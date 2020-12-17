//===-- Allocatable.cpp -- Allocatable statements lowering ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Allocatable.h"
#include "../runtime/allocatable.h"
#include "RTBuilder.h"
#include "StatementContext.h"
#include "flang/Evaluate/tools.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

/// Runtime call generators
using namespace Fortran::runtime;
static void genAllocatableSetBounds(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value boxAddress,
                                    mlir::Value dimIndex, mlir::Value lowerBoud,
                                    mlir::Value upperBound) {
  auto callee = Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableSetBounds)>(
      loc, builder);
  llvm::SmallVector<mlir::Value, 4> args{boxAddress, dimIndex, lowerBoud,
                                         upperBound};
  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  builder.create<fir::CallOp>(loc, callee, operands);
}

static void genAllocatableInitCharRtCall(Fortran::lower::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         fir::MutableBoxValue boxAddress,
                                         mlir::Value len) {
  auto callee =
      Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableInitCharacter)>(
          loc, builder);
  auto inputTypes = callee.getType().getInputs();
  if (inputTypes.size() != 5) {
    mlir::emitError(
        loc, "AllocatableInitCharacter runtime interface not as expected");
    return;
  }
  llvm::SmallVector<mlir::Value, 5> args;
  args.push_back(
      builder.createConvert(loc, inputTypes[0], boxAddress.getAddr()));
  args.push_back(builder.createConvert(loc, inputTypes[1], len));
  auto kind = boxAddress.getEleTy().cast<fir::CharacterType>().getFKind();
  args.push_back(builder.createIntegerConstant(loc, inputTypes[2], kind));
  auto rank = boxAddress.rank();
  args.push_back(builder.createIntegerConstant(loc, inputTypes[3], rank));
  // TODO: coarrays
  auto corank = 0;
  args.push_back(builder.createIntegerConstant(loc, inputTypes[4], corank));
  builder.create<fir::CallOp>(loc, callee, args);
}

static mlir::Value
genAllocatableAllocate(Fortran::lower::FirOpBuilder &builder,
                       mlir::Location loc, mlir::Value boxAddress,
                       mlir::Value hasStat, mlir::Value errMsgBox,
                       mlir::Value sourceFile, mlir::Value sourceLine) {
  auto callee = Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableAllocate)>(
      loc, builder);
  llvm::SmallVector<mlir::Value, 5> args{boxAddress, hasStat, errMsgBox,
                                         sourceFile, sourceLine};
  llvm::SmallVector<mlir::Value, 5> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  return builder.create<fir::CallOp>(loc, callee, operands).getResult(0);
}

static mlir::Value
genAllocatableDeallocate(Fortran::lower::FirOpBuilder &builder,
                         mlir::Location loc, mlir::Value boxAddress,
                         mlir::Value hasStat, mlir::Value errMsgBox,
                         mlir::Value sourceFile, mlir::Value sourceLine) {
  auto callee = Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableDeallocate)>(
      loc, builder);
  llvm::SmallVector<mlir::Value, 5> args{boxAddress, hasStat, errMsgBox,
                                         sourceFile, sourceLine};
  llvm::SmallVector<mlir::Value, 5> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  return builder.create<fir::CallOp>(loc, callee, operands).getResult(0);
}

/// Helper to get symbol from AllocateObject.
static const Fortran::semantics::Symbol &
unwrapSymbol(const Fortran::parser::AllocateObject &allocObj) {
  const auto &lastName = Fortran::parser::GetLastName(allocObj);
  assert(lastName.symbol);
  return *lastName.symbol;
}

// TODO: the front-end needs to store the AllocateObject as an expressions.
// When derived type are supported, the allocatable can be describe by a non
// trivial expression that would need to be computed e.g `A(foo(B+C),
// 1)%alloc_component` For now, getting the last name symbol is OK since there
// is only one name.
static fir::MutableBoxValue
genMutableBoxValue(Fortran::lower::AbstractConverter &converter,
                   mlir::Location loc,
                   const Fortran::parser::AllocateObject &allocObj) {
  const auto &symbol = unwrapSymbol(allocObj);
  Fortran::evaluate::DataRef ref(symbol);
  auto dyType = Fortran::evaluate::DynamicType::From(symbol);
  if (dyType)
    if (auto maybeExpr =
            Fortran::evaluate::TypedWrapper<Fortran::evaluate::Designator,
                                            Fortran::evaluate::DataRef>(
                *dyType, std::move(ref)))
      return converter.genExprMutableBox(*maybeExpr, &loc);
  fir::emitFatalError(
      loc, "could not build expression from symbol in allocate statement");
}

namespace {
// Lower ALLOCATE/DEALLOCATE stmt ERROR and STAT variable as well as the source
// file location to be passed to the runtime.
struct ErrorManagementValues {
  void lower(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
             const Fortran::lower::SomeExpr *statExpr,
             const ::Fortran::lower::SomeExpr *errMsgExpr) {
    auto builder = converter.getFirOpBuilder();
    if (statExpr) {
      TODO("lower stat expr in allocate and deallocate");
      hasStat = builder.createBool(loc, true);
    } else {
      hasStat = builder.createBool(loc, false);
    }

    if (errMsgExpr)
      TODO("errmsg in allocate and deallocate");
    else
      errMsgBoxAddr = builder.createNullConstant(loc);
    sourceFile = converter.locationToFilename(loc);
    sourceLine = converter.locationToLineNo(loc, builder.getIntegerType(32));
  }
  bool hasErrorRecovery() const { return static_cast<bool>(statAddr); }
  // Values always initialized before lowering individual allocations
  mlir::Value sourceLine;
  mlir::Value sourceFile;
  mlir::Value hasStat;
  mlir::Value errMsgBoxAddr;
  // Value created only in certain cases before lowering individual allocations
  mlir::Value statAddr;
};

class AllocateStmtHelper {
public:
  AllocateStmtHelper(Fortran::lower::AbstractConverter &converter,
                     const Fortran::parser::AllocateStmt &stmt,
                     mlir::Location loc)
      : converter{converter}, builder{converter.getFirOpBuilder()}, stmt{stmt},
        loc{loc} {}

  void lower() {
    visitAllocateOptions();
    lowerAllocateLengthParameters();
    errorManagement.lower(converter, loc, statExpr, errMsgExpr);
    // Create a landing block after all allocations so that
    // we can jump there in case of error.
    if (errorManagement.hasErrorRecovery())
      TODO("error recovery");

    // TODO lower source and mold.
    if (sourceExpr || moldExpr)
      TODO("lower MOLD/SOURCE expr in allocate");

    for (const auto &allocation :
         std::get<std::list<Fortran::parser::Allocation>>(stmt.t))
      lowerAllocation(unwrapAllocation(allocation));
  }

private:
  struct Allocation {
    const Fortran::parser::Allocation &alloc;
    const Fortran::semantics::DeclTypeSpec &type;
    bool hasCoarraySpec() const {
      return std::get<std::optional<Fortran::parser::AllocateCoarraySpec>>(
                 alloc.t)
          .has_value();
    }
    const auto &getAllocObj() const {
      return std::get<Fortran::parser::AllocateObject>(alloc.t);
    }
    const Fortran::semantics::Symbol &getSymbol() const {
      return unwrapSymbol(getAllocObj());
    }
    const auto &getShapeSpecs() const {
      return std::get<std::list<Fortran::parser::AllocateShapeSpec>>(alloc.t);
    }
  };

  Allocation unwrapAllocation(const Fortran::parser::Allocation &alloc) {
    const auto &allocObj = std::get<Fortran::parser::AllocateObject>(alloc.t);
    const auto &symbol = unwrapSymbol(allocObj);
    assert(symbol.GetType());
    return Allocation{alloc, *symbol.GetType()};
  }

  void visitAllocateOptions() {
    for (const auto &allocOption :
         std::get<std::list<Fortran::parser::AllocOpt>>(stmt.t))
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::StatOrErrmsg &statOrErr) {
                std::visit(
                    Fortran::common::visitors{
                        [&](const Fortran::parser::StatVariable &statVar) {
                          statExpr = Fortran::semantics::GetExpr(statVar);
                        },
                        [&](const Fortran::parser::MsgVariable &errMsgVar) {
                          errMsgExpr = Fortran::semantics::GetExpr(errMsgVar);
                        },
                    },
                    statOrErr.u);
              },
              [&](const Fortran::parser::AllocOpt::Source &source) {
                sourceExpr = Fortran::semantics::GetExpr(source.v.value());
              },
              [&](const Fortran::parser::AllocOpt::Mold &mold) {
                moldExpr = Fortran::semantics::GetExpr(mold.v.value());
              },
          },
          allocOption.u);
  }

  void lowerAllocation(const Allocation &alloc) {
    auto boxAddr = genMutableBoxValue(converter, loc, alloc.getAllocObj());
    mlir::Value backupBox;

    if (sourceExpr) {
      genSourceAllocation(alloc, boxAddr);
    } else if (moldExpr) {
      genMoldAllocation(alloc, boxAddr);
    } else {
      genSimpleAllocation(alloc, boxAddr);
    }

    if (errorManagement.hasErrorRecovery())
      handleError();
  }

  void handleError() {
    // Ensure allocation status was not modified and create jump to end
    // on allocate statement in case an error was met.
    TODO("Error hanlding in allocate statement");
  }

  static bool lowerBoundsAreOnes(const Allocation &alloc) {
    for (const auto &shapeSpec : alloc.getShapeSpecs())
      if (std::get<0>(shapeSpec.t))
        return false;
    return true;
  }

  /// Build name for the fir::allocmem generated for alloc.
  std::string mangleAlloc(const Allocation &alloc) {
    return converter.mangleName(alloc.getSymbol()) + ".alloc";
  }

  /// Generate allocation without runtime calls.
  /// Only for intrinsic types. No coarrays, no polymorphism. No error recovery.
  void genInlinedAllocation(const Allocation &alloc,
                            fir::MutableBoxValue &boxAddr) {
    mlir::SmallVector<mlir::Value, 2> lBounds;
    mlir::SmallVector<mlir::Value, 2> uBounds;
    Fortran::lower::StatementContext stmtCtx;
    auto idxTy = builder.getIndexType();
    auto lBoundsAreOnes = lowerBoundsAreOnes(alloc);
    for (const auto &shapeSpec : alloc.getShapeSpecs()) {
      if (!lBoundsAreOnes) {
        if (const auto &lbExpr = std::get<0>(shapeSpec.t)) {
          auto lb = fir::getBase(converter.genExprValue(
              Fortran::semantics::GetExpr(*lbExpr), stmtCtx, loc));
          lBounds.emplace_back(builder.createConvert(loc, idxTy, lb));
        } else {
          lBounds.emplace_back(builder.createIntegerConstant(loc, idxTy, 1));
        }
      }
      auto ub = fir::getBase(converter.genExprValue(
          Fortran::semantics::GetExpr(std::get<1>(shapeSpec.t)), stmtCtx, loc));
      uBounds.emplace_back(builder.createConvert(loc, idxTy, ub));
    }

    mlir::Value shape;
    mlir::SmallVector<mlir::Value, 2> extents;
    if (!uBounds.empty()) {
      if (lBoundsAreOnes) {
        auto shapeType =
            fir::ShapeType::get(builder.getContext(), uBounds.size());
        shape = builder.create<fir::ShapeOp>(loc, shapeType, uBounds);
        extents = uBounds;
      } else {
        auto one = builder.createIntegerConstant(loc, idxTy, 1);
        mlir::SmallVector<mlir::Value, 2> shapeShiftBounds;
        for (auto [lb, ub] : llvm::zip(lBounds, uBounds)) {
          auto diff = builder.create<mlir::SubIOp>(loc, ub, lb);
          auto extent = builder.create<mlir::AddIOp>(loc, diff, one);
          extents.emplace_back(extent);
          shapeShiftBounds.emplace_back(lb);
          shapeShiftBounds.emplace_back(extent);
        }
        auto shapeShiftType =
            fir::ShapeShiftType::get(builder.getContext(), uBounds.size());
        shape = builder.create<fir::ShapeShiftOp>(loc, shapeShiftType,
                                                  shapeShiftBounds);
      }
    }
    mlir::SmallVector<mlir::Value, 2> lengths;
    if (auto charTy = boxAddr.getEleTy().dyn_cast<fir::CharacterType>()) {
      if (charTy.getLen() == fir::CharacterType::unknownLen()) {
        if (!lenParams.empty())
          lengths.emplace_back(builder.createConvert(loc, idxTy, lenParams[0]));
        else if (boxAddr.hasNonDeferredLenParams())
          lengths.emplace_back(builder.createConvert(
              loc, idxTy, boxAddr.nonDeferredLenParams()[0]));
        else
          mlir::emitError(
              loc,
              "could not deduce character lengths in character allocation");
      }
    }

    // FIXME AllocMemOp is ignoring its length arguments. Squeezed in into the
    // extents for now.
    extents.append(lengths.begin(), lengths.end());
    mlir::Value heap = builder.create<fir::AllocMemOp>(
        loc, boxAddr.getBaseTy(), mangleAlloc(alloc), llvm::None, extents);
    mlir::Value emptySlice;
    auto box = builder.create<fir::EmboxOp>(loc, boxAddr.getBoxTy(), heap,
                                            shape, emptySlice, lengths);
    builder.create<fir::StoreOp>(loc, box, boxAddr.getAddr());
  }

  void genSimpleAllocation(const Allocation &alloc,
                           fir::MutableBoxValue boxAddr) {
    if (!boxAddr.isDerived() && !errorManagement.hasErrorRecovery() &&
        !alloc.type.IsPolymorphic() && !alloc.hasCoarraySpec()) {
      genInlinedAllocation(alloc, boxAddr);
      return;
    }
    if (alloc.hasCoarraySpec())
      TODO("coarray allocation");
    if (alloc.type.IsPolymorphic())
      genSetType(alloc, boxAddr);
    genSetDeferredLengthParameters(alloc, boxAddr);
    // Set bounds for arrays
    auto idxTy = builder.getIndexType();
    auto i32Ty = builder.getIntegerType(32);
    Fortran::lower::StatementContext stmtCtx;
    auto addr = boxAddr.getAddr();
    for (const auto &iter : llvm::enumerate(alloc.getShapeSpecs())) {
      mlir::Value lb;
      const auto &bounds = iter.value().t;
      if (const auto &lbExpr = std::get<0>(bounds))
        lb = fir::getBase(converter.genExprValue(
            Fortran::semantics::GetExpr(*lbExpr), stmtCtx, loc));
      else
        lb = builder.createIntegerConstant(loc, idxTy, 1);
      auto ub = fir::getBase(converter.genExprValue(
          Fortran::semantics::GetExpr(std::get<1>(bounds)), stmtCtx, loc));
      auto dimIndex = builder.createIntegerConstant(loc, i32Ty, iter.index());
      // Runtime call
      genAllocatableSetBounds(builder, loc, addr, dimIndex, lb, ub);
    }
    // Runtime call
    auto stat = genAllocatableAllocate(builder, loc, addr, getHasStat(),
                                       getErrMsgBoxAddr(), getSourceFile(),
                                       getSourceLine());
    if (auto statAddr = getStatAddr()) {
      auto castStat = builder.createConvert(
          loc, fir::dyn_cast_ptrEleTy(statAddr.getType()), stat);
      builder.create<fir::StoreOp>(loc, castStat, statAddr);
    }
  }

  /// Lower the length parameters that may be specified in the optional
  /// type specification.
  void lowerAllocateLengthParameters() {
    const auto *typeSpec = getAllocateStmtTypeSpec();
    if (!typeSpec)
      return;
    if (typeSpec->AsDerived()) {
      mlir::emitError(loc, "TODO: setting derived type params in allocation");
      return;
    }
    if (typeSpec->category() ==
        Fortran::semantics::DeclTypeSpec::Category::Character) {
      auto lenParam = typeSpec->characterTypeSpec().length();
      if (auto intExpr = lenParam.GetExplicit()) {
        Fortran::lower::StatementContext stmtCtx;
        Fortran::semantics::SomeExpr lenExpr{*intExpr};
        lenParams.push_back(
            fir::getBase(converter.genExprValue(lenExpr, stmtCtx, &loc)));
      }
    }
  }

  // Set length parameters in the box stored in boxAddr.
  // This must be called before setting the bounds because it may use
  // Init runtime calls that may set the bounds to zero.
  void genSetDeferredLengthParameters(const Allocation &alloc,
                                      fir::MutableBoxValue boxAddr) {
    if (lenParams.empty())
      return;
    // TODO: in case a length parameter was not deferred, insert a runtime check
    // that the length is the same (AllocatableCheckLengthParameter runtime
    // call).
    if (boxAddr.isCharacter())
      genAllocatableInitCharRtCall(builder, loc, boxAddr, lenParams[0]);
    // TODO: derived type
  }

  void genSourceAllocation(const Allocation &alloc,
                           fir::MutableBoxValue boxAddr) {
    TODO("SOURCE allocation lowering");
  }
  void genMoldAllocation(const Allocation &alloc,
                         fir::MutableBoxValue boxAddr) {
    TODO("MOLD allocation lowering");
  }
  void genSetType(const Allocation &alloc, fir::MutableBoxValue boxAddr) {
    TODO("Polymorphic entity allocation lowering");
  }

  mlir::Value getSourceLine() const {
    assert(errorManagement.sourceLine && "always needs to be lowered");
    return errorManagement.sourceLine;
  }
  mlir::Value getSourceFile() const {
    assert(errorManagement.sourceFile && "always needs to be lowered");
    return errorManagement.sourceFile;
  }
  mlir::Value getHasStat() {
    assert(errorManagement.sourceFile && "always needs to be lowered");
    return errorManagement.hasStat;
  }
  mlir::Value getErrMsgBoxAddr() {
    assert(errorManagement.sourceFile && "always needs to be lowered");
    return errorManagement.errMsgBoxAddr;
  }
  mlir::Value getStatAddr() const { return errorManagement.statAddr; }

  const Fortran::semantics::DeclTypeSpec *getAllocateStmtTypeSpec() const {
    if (const auto &typeSpec =
            std::get<std::optional<Fortran::parser::TypeSpec>>(stmt.t))
      return typeSpec->declTypeSpec;
    return nullptr;
  }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::FirOpBuilder &builder;
  const Fortran::parser::AllocateStmt &stmt;
  const Fortran::lower::SomeExpr *sourceExpr{nullptr};
  const Fortran::lower::SomeExpr *moldExpr{nullptr};
  const Fortran::lower::SomeExpr *statExpr{nullptr};
  const Fortran::lower::SomeExpr *errMsgExpr{nullptr};
  // If the allocate has a type spec, lenParams contains the
  // value of the length parameters that were specified inside.
  llvm::SmallVector<mlir::Value, 2> lenParams;
  ErrorManagementValues errorManagement;

  mlir::Location loc;
};
} // namespace

void Fortran::lower::genAllocateStmt(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::AllocateStmt &stmt, mlir::Location loc) {
  AllocateStmtHelper{converter, stmt, loc}.lower();
  return;
}

// Generate deallocation of a pointer/allocatable.
static void genDeallocate(Fortran::lower::FirOpBuilder &builder,
                          mlir::Location loc, fir::MutableBoxValue box,
                          ErrorManagementValues &errorManagement) {
  // For derived and when error recovery is present, use runtime.
  if (box.isDerived() || errorManagement.hasErrorRecovery()) {
    // TODO use return stat for error recovery
    genAllocatableDeallocate(
        builder, loc, box.getAddr(), errorManagement.hasStat,
        errorManagement.errMsgBoxAddr, errorManagement.sourceFile,
        errorManagement.sourceLine);
    return;
  }
  // Inlined deallocate.
  auto boxValue = builder.create<fir::LoadOp>(loc, box.getAddr());
  auto boxTy = box.getBoxTy();
  auto heapOrPtrTy = boxTy.getEleTy();
  auto addr = builder.create<fir::BoxAddrOp>(loc, heapOrPtrTy, boxValue);
  builder.create<fir::FreeMemOp>(loc, addr);
  auto deallocatedBox =
      createUnallocatedBox(builder, loc, boxTy, box.nonDeferredLenParams());
  builder.create<fir::StoreOp>(loc, deallocatedBox, box.getAddr());
}

void Fortran::lower::genDeallocateStmt(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::DeallocateStmt &stmt, mlir::Location loc) {
  const Fortran::lower::SomeExpr *statExpr{nullptr};
  const Fortran::lower::SomeExpr *errMsgExpr{nullptr};
  for (const auto &statOrErr :
       std::get<std::list<Fortran::parser::StatOrErrmsg>>(stmt.t))
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::parser::StatVariable &statVar) {
                     statExpr = Fortran::semantics::GetExpr(statVar);
                   },
                   [&](const Fortran::parser::MsgVariable &errMsgVar) {
                     errMsgExpr = Fortran::semantics::GetExpr(errMsgVar);
                   },
               },
               statOrErr.u);
  if (statExpr || errMsgExpr)
    TODO("error recovery in deallocate");
  ErrorManagementValues errorManagement;
  auto &builder = converter.getFirOpBuilder();
  errorManagement.lower(converter, loc, statExpr, errMsgExpr);
  for (const auto &allocateObject :
       std::get<std::list<Fortran::parser::AllocateObject>>(stmt.t)) {
    auto box = genMutableBoxValue(converter, loc, allocateObject);
    genDeallocate(builder, loc, box, errorManagement);
  }
}

mlir::Value
Fortran::lower::createUnallocatedBox(Fortran::lower::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Type boxType,
                                     mlir::ValueRange nonDeferredParams) {
  auto heapType = boxType.dyn_cast<fir::BoxType>().getEleTy();
  auto type = fir::dyn_cast_ptrEleTy(heapType);
  auto eleTy = type;
  if (auto seqType = eleTy.dyn_cast<fir::SequenceType>())
    eleTy = seqType.getEleTy();
  if (eleTy.isa<fir::RecordType>())
    mlir::emitError(loc, "TODO: Derived type allocatable initialization");
  auto nullAddr = builder.createNullConstant(loc, heapType);
  mlir::Value shape;
  if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
    auto zero = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
    llvm::SmallVector<mlir::Value, 2> extents(seqTy.getDimension(), zero);
    llvm::ArrayRef<mlir::Value> lbounds = llvm::None;
    shape = builder.createShape(loc,
                                fir::ArrayBoxValue{nullAddr, extents, lbounds});
  }
  // Provide dummy length parameters if they are dynamic. If a length parameter
  // is deferred. it is set to zero here and will be set on allocation.
  llvm::SmallVector<mlir::Value, 2> lenParams;
  if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
    if (charTy.getLen() == fir::CharacterType::unknownLen()) {
      if (!nonDeferredParams.empty()) {
        lenParams.push_back(nonDeferredParams[0]);
      } else {
        auto zero =
            builder.createIntegerConstant(loc, builder.getIndexType(), 0);
        lenParams.push_back(zero);
      }
    }
  }
  mlir::Value emptySlice;
  return builder.create<fir::EmboxOp>(loc, boxType, nullAddr, shape, emptySlice,
                                      lenParams);
}
