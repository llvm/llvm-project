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
#include "flang/Evaluate/tools.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/IterationSpace.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/pointer.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "llvm/Support/CommandLine.h"

/// By default fir memory operation fir::AllocMemOp/fir::FreeMemOp are used.
/// This switch allow forcing the use of runtime and descriptors for everything.
/// This is mainly intended as a debug switch.
static llvm::cl::opt<bool> useAllocateRuntime(
    "use-alloc-runtime",
    llvm::cl::desc("Lower allocations to fortran runtime calls"),
    llvm::cl::init(false));
/// Switch to force lowering of allocatable and pointers to descriptors in all
/// cases for debug purposes.
static llvm::cl::opt<bool> useDescForMutableBox(
    "use-desc-for-alloc",
    llvm::cl::desc("Always use descriptors for POINTER and ALLOCATABLE"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Error management
//===----------------------------------------------------------------------===//

namespace {
// Manage STAT and ERRMSG specifier information across a sequence of runtime
// calls for an ALLOCATE/DEALLOCATE stmt.
struct ErrorManager {
  void init(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
            const Fortran::lower::SomeExpr *statExpr,
            const Fortran::lower::SomeExpr *errMsgExpr) {
    Fortran::lower::StatementContext stmtCtx;
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    hasStat = builder.createBool(loc, statExpr != nullptr);
    statAddr = statExpr
                   ? fir::getBase(converter.genExprAddr(loc, statExpr, stmtCtx))
                   : mlir::Value{};
    errMsgAddr =
        statExpr && errMsgExpr
            ? builder.createBox(loc,
                                converter.genExprAddr(loc, errMsgExpr, stmtCtx))
            : builder.create<fir::AbsentOp>(
                  loc,
                  fir::BoxType::get(mlir::NoneType::get(builder.getContext())));
    sourceFile = fir::factory::locationToFilename(builder, loc);
    sourceLine = fir::factory::locationToLineNo(builder, loc,
                                                builder.getIntegerType(32));
  }

  bool hasStatSpec() const { return static_cast<bool>(statAddr); }

  void genStatCheck(fir::FirOpBuilder &builder, mlir::Location loc) {
    if (statValue) {
      mlir::Value zero =
          builder.createIntegerConstant(loc, statValue.getType(), 0);
      auto cmp = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, statValue, zero);
      auto ifOp = builder.create<fir::IfOp>(loc, cmp,
                                            /*withElseRegion=*/false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }
  }

  void assignStat(fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::Value stat) {
    if (hasStatSpec()) {
      assert(stat && "missing stat value");
      mlir::Value castStat = builder.createConvert(
          loc, fir::dyn_cast_ptrEleTy(statAddr.getType()), stat);
      builder.create<fir::StoreOp>(loc, castStat, statAddr);
      statValue = stat;
    }
  }

  mlir::Value hasStat;
  mlir::Value errMsgAddr;
  mlir::Value sourceFile;
  mlir::Value sourceLine;

private:
  mlir::Value statAddr;  // STAT variable address
  mlir::Value statValue; // current runtime STAT value
};

//===----------------------------------------------------------------------===//
// Allocatables runtime call generators
//===----------------------------------------------------------------------===//

using namespace Fortran::runtime;
/// Generate a runtime call to set the bounds of an allocatable or pointer
/// descriptor.
static void genRuntimeSetBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                                const fir::MutableBoxValue &box,
                                mlir::Value dimIndex, mlir::Value lowerBound,
                                mlir::Value upperBound) {
  mlir::func::FuncOp callee =
      box.isPointer()
          ? fir::runtime::getRuntimeFunc<mkRTKey(PointerSetBounds)>(loc,
                                                                    builder)
          : fir::runtime::getRuntimeFunc<mkRTKey(AllocatableSetBounds)>(
                loc, builder);
  llvm::SmallVector<mlir::Value> args{box.getAddr(), dimIndex, lowerBound,
                                      upperBound};
  llvm::SmallVector<mlir::Value> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getFunctionType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  builder.create<fir::CallOp>(loc, callee, operands);
}

/// Generate runtime call to set the lengths of a character allocatable or
/// pointer descriptor.
static void genRuntimeInitCharacter(fir::FirOpBuilder &builder,
                                    mlir::Location loc,
                                    const fir::MutableBoxValue &box,
                                    mlir::Value len) {
  mlir::func::FuncOp callee =
      box.isPointer()
          ? fir::runtime::getRuntimeFunc<mkRTKey(PointerNullifyCharacter)>(
                loc, builder)
          : fir::runtime::getRuntimeFunc<mkRTKey(AllocatableInitCharacter)>(
                loc, builder);
  llvm::ArrayRef<mlir::Type> inputTypes = callee.getFunctionType().getInputs();
  if (inputTypes.size() != 5)
    fir::emitFatalError(
        loc, "AllocatableInitCharacter runtime interface not as expected");
  llvm::SmallVector<mlir::Value> args;
  args.push_back(builder.createConvert(loc, inputTypes[0], box.getAddr()));
  args.push_back(builder.createConvert(loc, inputTypes[1], len));
  int kind = box.getEleTy().cast<fir::CharacterType>().getFKind();
  args.push_back(builder.createIntegerConstant(loc, inputTypes[2], kind));
  int rank = box.rank();
  args.push_back(builder.createIntegerConstant(loc, inputTypes[3], rank));
  // TODO: coarrays
  int corank = 0;
  args.push_back(builder.createIntegerConstant(loc, inputTypes[4], corank));
  builder.create<fir::CallOp>(loc, callee, args);
}

/// Generate a sequence of runtime calls to allocate memory.
static mlir::Value genRuntimeAllocate(fir::FirOpBuilder &builder,
                                      mlir::Location loc,
                                      const fir::MutableBoxValue &box,
                                      ErrorManager &errorManager) {
  mlir::func::FuncOp callee =
      box.isPointer()
          ? fir::runtime::getRuntimeFunc<mkRTKey(PointerAllocate)>(loc, builder)
          : fir::runtime::getRuntimeFunc<mkRTKey(AllocatableAllocate)>(loc,
                                                                       builder);
  llvm::SmallVector<mlir::Value> args{
      box.getAddr(), errorManager.hasStat, errorManager.errMsgAddr,
      errorManager.sourceFile, errorManager.sourceLine};
  llvm::SmallVector<mlir::Value> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getFunctionType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  return builder.create<fir::CallOp>(loc, callee, operands).getResult(0);
}

/// Generate a sequence of runtime calls to allocate memory and assign with the
/// \p source.
static mlir::Value genRuntimeAllocateSource(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            const fir::MutableBoxValue &box,
                                            fir::ExtendedValue source,
                                            ErrorManager &errorManager) {
  mlir::func::FuncOp callee =
      box.isPointer()
          ? fir::runtime::getRuntimeFunc<mkRTKey(PointerAllocateSource)>(
                loc, builder)
          : fir::runtime::getRuntimeFunc<mkRTKey(AllocatableAllocateSource)>(
                loc, builder);
  llvm::SmallVector<mlir::Value> args{
      box.getAddr(),           fir::getBase(source),
      errorManager.hasStat,    errorManager.errMsgAddr,
      errorManager.sourceFile, errorManager.sourceLine};
  llvm::SmallVector<mlir::Value> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getFunctionType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  return builder.create<fir::CallOp>(loc, callee, operands).getResult(0);
}

/// Generate a runtime call to deallocate memory.
static mlir::Value genRuntimeDeallocate(fir::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        const fir::MutableBoxValue &box,
                                        ErrorManager &errorManager,
                                        mlir::Value declaredTypeDesc = {}) {
  // Ensure fir.box is up-to-date before passing it to deallocate runtime.
  mlir::Value boxAddress = fir::factory::getMutableIRBox(builder, loc, box);
  mlir::func::FuncOp callee;
  llvm::SmallVector<mlir::Value> args;
  llvm::SmallVector<mlir::Value> operands;
  if (box.isPolymorphic() || box.isUnlimitedPolymorphic()) {
    callee = box.isPointer()
                 ? fir::runtime::getRuntimeFunc<mkRTKey(
                       PointerDeallocatePolymorphic)>(loc, builder)
                 : fir::runtime::getRuntimeFunc<mkRTKey(
                       AllocatableDeallocatePolymorphic)>(loc, builder);
    if (!declaredTypeDesc)
      declaredTypeDesc = builder.createNullConstant(loc);
    operands = fir::runtime::createArguments(
        builder, loc, callee.getFunctionType(), boxAddress, declaredTypeDesc,
        errorManager.hasStat, errorManager.errMsgAddr, errorManager.sourceFile,
        errorManager.sourceLine);
  } else {
    callee = box.isPointer()
                 ? fir::runtime::getRuntimeFunc<mkRTKey(PointerDeallocate)>(
                       loc, builder)
                 : fir::runtime::getRuntimeFunc<mkRTKey(AllocatableDeallocate)>(
                       loc, builder);
    operands = fir::runtime::createArguments(
        builder, loc, callee.getFunctionType(), boxAddress,
        errorManager.hasStat, errorManager.errMsgAddr, errorManager.sourceFile,
        errorManager.sourceLine);
  }
  return builder.create<fir::CallOp>(loc, callee, operands).getResult(0);
}

//===----------------------------------------------------------------------===//
// Allocate statement implementation
//===----------------------------------------------------------------------===//

/// Helper to get symbol from AllocateObject.
static const Fortran::semantics::Symbol &
unwrapSymbol(const Fortran::parser::AllocateObject &allocObj) {
  const Fortran::parser::Name &lastName =
      Fortran::parser::GetLastName(allocObj);
  assert(lastName.symbol);
  return *lastName.symbol;
}

static fir::MutableBoxValue
genMutableBoxValue(Fortran::lower::AbstractConverter &converter,
                   mlir::Location loc,
                   const Fortran::parser::AllocateObject &allocObj) {
  const Fortran::lower::SomeExpr *expr = Fortran::semantics::GetExpr(allocObj);
  assert(expr && "semantic analysis failure");
  return converter.genExprMutableBox(loc, *expr);
}

/// Implement Allocate statement lowering.
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
    errorManager.init(converter, loc, statExpr, errMsgExpr);
    Fortran::lower::StatementContext stmtCtx;
    if (sourceExpr)
      sourceExv = converter.genExprBox(loc, *sourceExpr, stmtCtx);
    if (moldExpr)
      TODO(loc, "lower MOLD expr in allocate");
    mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();
    for (const auto &allocation :
         std::get<std::list<Fortran::parser::Allocation>>(stmt.t))
      lowerAllocation(unwrapAllocation(allocation));
    builder.restoreInsertionPoint(insertPt);
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
    const Fortran::parser::AllocateObject &getAllocObj() const {
      return std::get<Fortran::parser::AllocateObject>(alloc.t);
    }
    const Fortran::semantics::Symbol &getSymbol() const {
      return unwrapSymbol(getAllocObj());
    }
    const std::list<Fortran::parser::AllocateShapeSpec> &getShapeSpecs() const {
      return std::get<std::list<Fortran::parser::AllocateShapeSpec>>(alloc.t);
    }
  };

  Allocation unwrapAllocation(const Fortran::parser::Allocation &alloc) {
    const auto &allocObj = std::get<Fortran::parser::AllocateObject>(alloc.t);
    const Fortran::semantics::Symbol &symbol = unwrapSymbol(allocObj);
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
    fir::MutableBoxValue boxAddr =
        genMutableBoxValue(converter, loc, alloc.getAllocObj());

    if (sourceExpr) {
      genSourceAllocation(alloc, boxAddr);
    } else if (moldExpr) {
      genMoldAllocation(alloc, boxAddr);
    } else {
      genSimpleAllocation(alloc, boxAddr);
    }
  }

  static bool lowerBoundsAreOnes(const Allocation &alloc) {
    for (const Fortran::parser::AllocateShapeSpec &shapeSpec :
         alloc.getShapeSpecs())
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
                            const fir::MutableBoxValue &box) {
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    Fortran::lower::StatementContext stmtCtx;
    mlir::Type idxTy = builder.getIndexType();
    bool lBoundsAreOnes = lowerBoundsAreOnes(alloc);
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    for (const Fortran::parser::AllocateShapeSpec &shapeSpec :
         alloc.getShapeSpecs()) {
      mlir::Value lb;
      if (!lBoundsAreOnes) {
        if (const std::optional<Fortran::parser::BoundExpr> &lbExpr =
                std::get<0>(shapeSpec.t)) {
          lb = fir::getBase(converter.genExprValue(
              loc, Fortran::semantics::GetExpr(*lbExpr), stmtCtx));
          lb = builder.createConvert(loc, idxTy, lb);
        } else {
          lb = one;
        }
        lbounds.emplace_back(lb);
      }
      mlir::Value ub = fir::getBase(converter.genExprValue(
          loc, Fortran::semantics::GetExpr(std::get<1>(shapeSpec.t)), stmtCtx));
      ub = builder.createConvert(loc, idxTy, ub);
      if (lb) {
        mlir::Value diff = builder.create<mlir::arith::SubIOp>(loc, ub, lb);
        extents.emplace_back(
            builder.create<mlir::arith::AddIOp>(loc, diff, one));
      } else {
        extents.emplace_back(ub);
      }
    }
    fir::factory::genInlinedAllocation(builder, loc, box, lbounds, extents,
                                       lenParams, mangleAlloc(alloc));
  }

  void genSimpleAllocation(const Allocation &alloc,
                           const fir::MutableBoxValue &box) {
    if (!box.isDerived() && !errorManager.hasStatSpec() &&
        !alloc.type.IsPolymorphic() && !alloc.hasCoarraySpec() &&
        !useAllocateRuntime) {
      genInlinedAllocation(alloc, box);
      return;
    }
    // Generate a sequence of runtime calls.
    errorManager.genStatCheck(builder, loc);
    genAllocateObjectInit(box);
    if (alloc.hasCoarraySpec())
      TODO(loc, "coarray allocation");
    if (alloc.type.IsPolymorphic())
      genSetType(alloc, box, loc);
    genSetDeferredLengthParameters(alloc, box);
    genAllocateObjectBounds(alloc, box);
    mlir::Value stat = genRuntimeAllocate(builder, loc, box, errorManager);
    fir::factory::syncMutableBoxFromIRBox(builder, loc, box);
    errorManager.assignStat(builder, loc, stat);
  }

  /// Lower the length parameters that may be specified in the optional
  /// type specification.
  void lowerAllocateLengthParameters() {
    const Fortran::semantics::DeclTypeSpec *typeSpec =
        getIfAllocateStmtTypeSpec();
    if (!typeSpec)
      return;
    if (const Fortran::semantics::DerivedTypeSpec *derived =
            typeSpec->AsDerived())
      if (Fortran::semantics::CountLenParameters(*derived) > 0)
        TODO(loc, "setting derived type params in allocation");
    if (typeSpec->category() ==
        Fortran::semantics::DeclTypeSpec::Category::Character) {
      Fortran::semantics::ParamValue lenParam =
          typeSpec->characterTypeSpec().length();
      if (Fortran::semantics::MaybeIntExpr intExpr = lenParam.GetExplicit()) {
        Fortran::lower::StatementContext stmtCtx;
        Fortran::lower::SomeExpr lenExpr{*intExpr};
        lenParams.push_back(
            fir::getBase(converter.genExprValue(loc, lenExpr, stmtCtx)));
      }
    }
  }

  // Set length parameters in the box stored in boxAddr.
  // This must be called before setting the bounds because it may use
  // Init runtime calls that may set the bounds to zero.
  void genSetDeferredLengthParameters(const Allocation &alloc,
                                      const fir::MutableBoxValue &box) {
    if (lenParams.empty())
      return;
    // TODO: in case a length parameter was not deferred, insert a runtime check
    // that the length is the same (AllocatableCheckLengthParameter runtime
    // call).
    if (box.isCharacter())
      genRuntimeInitCharacter(builder, loc, box, lenParams[0]);

    if (box.isDerived())
      TODO(loc, "derived type length parameters in allocate");
  }

  void genAllocateObjectInit(const fir::MutableBoxValue &box) {
    if (box.isPointer()) {
      // For pointers, the descriptor may still be uninitialized (see Fortran
      // 2018 19.5.2.2). The allocation runtime needs to be given a descriptor
      // with initialized rank, types and attributes. Initialize the descriptor
      // here to ensure these constraints are fulfilled.
      mlir::Value nullPointer = fir::factory::createUnallocatedBox(
          builder, loc, box.getBoxTy(), box.nonDeferredLenParams());
      builder.create<fir::StoreOp>(loc, nullPointer, box.getAddr());
    } else {
      assert(box.isAllocatable() && "must be an allocatable");
      // For allocatables, sync the MutableBoxValue and descriptor before the
      // calls in case it is tracked locally by a set of variables.
      fir::factory::getMutableIRBox(builder, loc, box);
    }
  }

  void genAllocateObjectBounds(const Allocation &alloc,
                               const fir::MutableBoxValue &box) {
    // Set bounds for arrays
    mlir::Type idxTy = builder.getIndexType();
    mlir::Type i32Ty = builder.getIntegerType(32);
    Fortran::lower::StatementContext stmtCtx;
    for (const auto &iter : llvm::enumerate(alloc.getShapeSpecs())) {
      mlir::Value lb;
      const auto &bounds = iter.value().t;
      if (const std::optional<Fortran::parser::BoundExpr> &lbExpr =
              std::get<0>(bounds))
        lb = fir::getBase(converter.genExprValue(
            loc, Fortran::semantics::GetExpr(*lbExpr), stmtCtx));
      else
        lb = builder.createIntegerConstant(loc, idxTy, 1);
      mlir::Value ub = fir::getBase(converter.genExprValue(
          loc, Fortran::semantics::GetExpr(std::get<1>(bounds)), stmtCtx));
      mlir::Value dimIndex =
          builder.createIntegerConstant(loc, i32Ty, iter.index());
      // Runtime call
      genRuntimeSetBounds(builder, loc, box, dimIndex, lb, ub);
    }
    if (sourceExpr && sourceExpr->Rank() > 0 &&
        alloc.getShapeSpecs().size() == 0) {
      // If the alloc object does not have shape list, get the bounds from the
      // source expression.
      mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
      const auto *sourceBox = sourceExv.getBoxOf<fir::BoxValue>();
      assert(sourceBox && "source expression should be lowered to one box");
      for (int i = 0; i < sourceExpr->Rank(); ++i) {
        auto dimVal = builder.createIntegerConstant(loc, idxTy, i);
        auto dimInfo = builder.create<fir::BoxDimsOp>(
            loc, idxTy, idxTy, idxTy, sourceBox->getAddr(), dimVal);
        mlir::Value lb =
            fir::factory::readLowerBound(builder, loc, sourceExv, i, one);
        mlir::Value extent = dimInfo.getResult(1);
        mlir::Value ub = builder.create<mlir::arith::SubIOp>(
            loc, builder.create<mlir::arith::AddIOp>(loc, extent, lb), one);
        mlir::Value dimIndex = builder.createIntegerConstant(loc, i32Ty, i);
        genRuntimeSetBounds(builder, loc, box, dimIndex, lb, ub);
      }
    }
  }

  void genSourceAllocation(const Allocation &alloc,
                           const fir::MutableBoxValue &box) {
    // Generate a sequence of runtime calls.
    errorManager.genStatCheck(builder, loc);
    genAllocateObjectInit(box);
    if (alloc.hasCoarraySpec())
      TODO(loc, "coarray allocation");
    if (alloc.type.IsPolymorphic())
      TODO(loc, "polymorphic allocation with SOURCE specifier");
    // Set length of the allocate object if it has. Otherwise, get the length
    // from source for the deferred length parameter.
    if (lenParams.empty() && box.isCharacter() &&
        !box.hasNonDeferredLenParams())
      lenParams.push_back(fir::factory::readCharLen(builder, loc, sourceExv));
    genSetDeferredLengthParameters(alloc, box);
    genAllocateObjectBounds(alloc, box);
    mlir::Value stat =
        genRuntimeAllocateSource(builder, loc, box, sourceExv, errorManager);
    fir::factory::syncMutableBoxFromIRBox(builder, loc, box);
    errorManager.assignStat(builder, loc, stat);
  }
  void genMoldAllocation(const Allocation &, const fir::MutableBoxValue &) {
    TODO(loc, "MOLD allocation");
  }

  /// Generate call to the AllocatableInitDerived to set up the type descriptor
  /// and other part of the descriptor for derived type.
  void genSetType(const Allocation &alloc, const fir::MutableBoxValue &box,
                  mlir::Location loc) {
    const Fortran::semantics::DeclTypeSpec *typeSpec =
        getIfAllocateStmtTypeSpec();

    // No type spec provided in allocate statement so the declared type spec is
    // used.
    if (!typeSpec)
      typeSpec = &alloc.type;
    assert(typeSpec && "type spec missing for polymorphic allocation");

    // Set up the descriptor for allocation for intrinsic type spec on
    // unlimited polymorphic entity.
    if (typeSpec->AsIntrinsic() &&
        fir::isUnlimitedPolymorphicType(fir::getBase(box).getType())) {
      mlir::func::FuncOp callee =
          box.isPointer()
              ? fir::runtime::getRuntimeFunc<mkRTKey(PointerNullifyIntrinsic)>(
                    loc, builder)
              : fir::runtime::getRuntimeFunc<mkRTKey(AllocatableInitIntrinsic)>(
                    loc, builder);

      llvm::ArrayRef<mlir::Type> inputTypes =
          callee.getFunctionType().getInputs();
      llvm::SmallVector<mlir::Value> args;
      args.push_back(builder.createConvert(loc, inputTypes[0], box.getAddr()));
      mlir::Value category = builder.createIntegerConstant(
          loc, inputTypes[1],
          static_cast<int32_t>(typeSpec->AsIntrinsic()->category()));
      mlir::Value kind = builder.createIntegerConstant(
          loc, inputTypes[2],
          Fortran::evaluate::ToInt64(typeSpec->AsIntrinsic()->kind()).value());
      mlir::Value rank = builder.createIntegerConstant(
          loc, inputTypes[3], alloc.getSymbol().Rank());
      mlir::Value corank = builder.createIntegerConstant(loc, inputTypes[4], 0);
      args.push_back(category);
      args.push_back(kind);
      args.push_back(rank);
      args.push_back(corank);
      builder.create<fir::CallOp>(loc, callee, args);
      return;
    }

    // Do not generate calls for non derived-type type spec.
    if (!typeSpec->AsDerived())
      return;

    auto typeDescAddr = Fortran::lower::getTypeDescAddr(
        builder, loc, typeSpec->derivedTypeSpec());
    mlir::func::FuncOp callee =
        box.isPointer()
            ? fir::runtime::getRuntimeFunc<mkRTKey(PointerNullifyDerived)>(
                  loc, builder)
            : fir::runtime::getRuntimeFunc<mkRTKey(AllocatableInitDerived)>(
                  loc, builder);

    llvm::ArrayRef<mlir::Type> inputTypes =
        callee.getFunctionType().getInputs();
    llvm::SmallVector<mlir::Value> args;
    args.push_back(builder.createConvert(loc, inputTypes[0], box.getAddr()));
    args.push_back(builder.createConvert(loc, inputTypes[1], typeDescAddr));
    mlir::Value rank = builder.createIntegerConstant(loc, inputTypes[2],
                                                     alloc.getSymbol().Rank());
    mlir::Value corank = builder.createIntegerConstant(loc, inputTypes[3], 0);
    args.push_back(rank);
    args.push_back(corank);
    builder.create<fir::CallOp>(loc, callee, args);
  }

  /// Returns a pointer to the DeclTypeSpec if a type-spec is provided in the
  /// allocate statement. Returns a null pointer otherwise.
  const Fortran::semantics::DeclTypeSpec *getIfAllocateStmtTypeSpec() const {
    if (const auto &typeSpec =
            std::get<std::optional<Fortran::parser::TypeSpec>>(stmt.t))
      return typeSpec->declTypeSpec;
    return nullptr;
  }

  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  const Fortran::parser::AllocateStmt &stmt;
  const Fortran::lower::SomeExpr *sourceExpr{nullptr};
  const Fortran::lower::SomeExpr *moldExpr{nullptr};
  const Fortran::lower::SomeExpr *statExpr{nullptr};
  const Fortran::lower::SomeExpr *errMsgExpr{nullptr};
  // If the allocate has a type spec, lenParams contains the
  // value of the length parameters that were specified inside.
  llvm::SmallVector<mlir::Value> lenParams;
  ErrorManager errorManager;
  // 9.7.1.2(7) The source-expr is evaluated exactly once for each AllocateStmt.
  fir::ExtendedValue sourceExv;

  mlir::Location loc;
};
} // namespace

void Fortran::lower::genAllocateStmt(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::AllocateStmt &stmt, mlir::Location loc) {
  AllocateStmtHelper{converter, stmt, loc}.lower();
}

//===----------------------------------------------------------------------===//
// Deallocate statement implementation
//===----------------------------------------------------------------------===//

// Generate deallocation of a pointer/allocatable.
static void genDeallocate(fir::FirOpBuilder &builder, mlir::Location loc,
                          const fir::MutableBoxValue &box,
                          ErrorManager &errorManager,
                          mlir::Value declaredTypeDesc = {}) {
  // Deallocate intrinsic types inline.
  if (!box.isDerived() && !box.isPolymorphic() &&
      !box.isUnlimitedPolymorphic() && !errorManager.hasStatSpec() &&
      !useAllocateRuntime) {
    fir::factory::genInlinedDeallocate(builder, loc, box);
    return;
  }
  // Use runtime calls to deallocate descriptor cases. Sync MutableBoxValue
  // with its descriptor before and after calls if needed.
  errorManager.genStatCheck(builder, loc);
  mlir::Value stat =
      genRuntimeDeallocate(builder, loc, box, errorManager, declaredTypeDesc);
  fir::factory::syncMutableBoxFromIRBox(builder, loc, box);
  errorManager.assignStat(builder, loc, stat);
}

void Fortran::lower::genDeallocateBox(
    Fortran::lower::AbstractConverter &converter,
    const fir::MutableBoxValue &box, mlir::Location loc,
    mlir::Value declaredTypeDesc) {
  const Fortran::lower::SomeExpr *statExpr = nullptr;
  const Fortran::lower::SomeExpr *errMsgExpr = nullptr;
  ErrorManager errorManager;
  errorManager.init(converter, loc, statExpr, errMsgExpr);
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  genDeallocate(builder, loc, box, errorManager, declaredTypeDesc);
}

void Fortran::lower::genDeallocateStmt(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::DeallocateStmt &stmt, mlir::Location loc) {
  const Fortran::lower::SomeExpr *statExpr = nullptr;
  const Fortran::lower::SomeExpr *errMsgExpr = nullptr;
  for (const Fortran::parser::StatOrErrmsg &statOrErr :
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
  ErrorManager errorManager;
  errorManager.init(converter, loc, statExpr, errMsgExpr);
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();
  for (const Fortran::parser::AllocateObject &allocateObject :
       std::get<std::list<Fortran::parser::AllocateObject>>(stmt.t)) {
    fir::MutableBoxValue box =
        genMutableBoxValue(converter, loc, allocateObject);

    mlir::Value declaredTypeDesc = {};
    if (box.isPolymorphic()) {
      const Fortran::semantics::Symbol &symbol = unwrapSymbol(allocateObject);
      assert(symbol.GetType());
      if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
              symbol.GetType()->AsDerived()) {
        declaredTypeDesc =
            Fortran::lower::getTypeDescAddr(builder, loc, *derivedTypeSpec);
      }
    }
    genDeallocate(builder, loc, box, errorManager, declaredTypeDesc);
  }
  builder.restoreInsertionPoint(insertPt);
}

//===----------------------------------------------------------------------===//
// MutableBoxValue creation implementation
//===----------------------------------------------------------------------===//

/// Is this symbol a pointer to a pointer array that does not have the
/// CONTIGUOUS attribute ?
static inline bool
isNonContiguousArrayPointer(const Fortran::semantics::Symbol &sym) {
  return Fortran::semantics::IsPointer(sym) && sym.Rank() != 0 &&
         !sym.attrs().test(Fortran::semantics::Attr::CONTIGUOUS);
}

/// Is this symbol a polymorphic pointer?
static inline bool isPolymorphicPointer(const Fortran::semantics::Symbol &sym) {
  return Fortran::semantics::IsPointer(sym) &&
         Fortran::semantics::IsPolymorphic(sym);
}

/// Is this symbol a polymorphic allocatable?
static inline bool
isPolymorphicAllocatable(const Fortran::semantics::Symbol &sym) {
  return Fortran::semantics::IsAllocatable(sym) &&
         Fortran::semantics::IsPolymorphic(sym);
}

/// Is this a local procedure symbol in a procedure that contains internal
/// procedures ?
static bool mayBeCapturedInInternalProc(const Fortran::semantics::Symbol &sym) {
  const Fortran::semantics::Scope &owner = sym.owner();
  Fortran::semantics::Scope::Kind kind = owner.kind();
  // Test if this is a procedure scope that contains a subprogram scope that is
  // not an interface.
  if (kind == Fortran::semantics::Scope::Kind::Subprogram ||
      kind == Fortran::semantics::Scope::Kind::MainProgram)
    for (const Fortran::semantics::Scope &childScope : owner.children())
      if (childScope.kind() == Fortran::semantics::Scope::Kind::Subprogram)
        if (const Fortran::semantics::Symbol *childSym = childScope.symbol())
          if (const auto *details =
                  childSym->detailsIf<Fortran::semantics::SubprogramDetails>())
            if (!details->isInterface())
              return true;
  return false;
}

/// In case it is safe to track the properties in variables outside a
/// descriptor, create the variables to hold the mutable properties of the
/// entity var. The variables are not initialized here.
static fir::MutableProperties
createMutableProperties(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc,
                        const Fortran::lower::pft::Variable &var,
                        mlir::ValueRange nonDeferredParams) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  // Globals and dummies may be associated, creating local variables would
  // require keeping the values and descriptor before and after every single
  // impure calls in the current scope (not only the ones taking the variable as
  // arguments. All.) Volatile means the variable may change in ways not defined
  // per Fortran, so lowering can most likely not keep the descriptor and values
  // in sync as needed.
  // Pointers to non contiguous arrays need to be represented with a fir.box to
  // account for the discontiguity.
  // Pointer/Allocatable in internal procedure are descriptors in the host link,
  // and it would increase complexity to sync this descriptor with the local
  // values every time the host link is escaping.
  if (var.isGlobal() || Fortran::semantics::IsDummy(sym) ||
      Fortran::semantics::IsFunctionResult(sym) ||
      sym.attrs().test(Fortran::semantics::Attr::VOLATILE) ||
      isNonContiguousArrayPointer(sym) || useAllocateRuntime ||
      useDescForMutableBox || mayBeCapturedInInternalProc(sym) ||
      isPolymorphicPointer(sym) || isPolymorphicAllocatable(sym))
    return {};
  fir::MutableProperties mutableProperties;
  std::string name = converter.mangleName(sym);
  mlir::Type baseAddrTy = converter.genType(sym);
  if (auto boxType = baseAddrTy.dyn_cast<fir::BaseBoxType>())
    baseAddrTy = boxType.getEleTy();
  // Allocate and set a variable to hold the address.
  // It will be set to null in setUnallocatedStatus.
  mutableProperties.addr = builder.allocateLocal(
      loc, baseAddrTy, name + ".addr", "",
      /*shape=*/std::nullopt, /*typeparams=*/std::nullopt);
  // Allocate variables to hold lower bounds and extents.
  int rank = sym.Rank();
  mlir::Type idxTy = builder.getIndexType();
  for (decltype(rank) i = 0; i < rank; ++i) {
    mlir::Value lboundVar = builder.allocateLocal(
        loc, idxTy, name + ".lb" + std::to_string(i), "",
        /*shape=*/std::nullopt, /*typeparams=*/std::nullopt);
    mlir::Value extentVar = builder.allocateLocal(
        loc, idxTy, name + ".ext" + std::to_string(i), "",
        /*shape=*/std::nullopt, /*typeparams=*/std::nullopt);
    mutableProperties.lbounds.emplace_back(lboundVar);
    mutableProperties.extents.emplace_back(extentVar);
  }

  // Allocate variable to hold deferred length parameters.
  mlir::Type eleTy = baseAddrTy;
  if (auto newTy = fir::dyn_cast_ptrEleTy(eleTy))
    eleTy = newTy;
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    eleTy = seqTy.getEleTy();
  if (auto record = eleTy.dyn_cast<fir::RecordType>())
    if (record.getNumLenParams() != 0)
      TODO(loc, "deferred length type parameters.");
  if (fir::isa_char(eleTy) && nonDeferredParams.empty()) {
    mlir::Value lenVar =
        builder.allocateLocal(loc, builder.getCharacterLengthType(),
                              name + ".len", "", /*shape=*/std::nullopt,
                              /*typeparams=*/std::nullopt);
    mutableProperties.deferredParams.emplace_back(lenVar);
  }
  return mutableProperties;
}

fir::MutableBoxValue Fortran::lower::createMutableBox(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::pft::Variable &var, mlir::Value boxAddr,
    mlir::ValueRange nonDeferredParams) {

  fir::MutableProperties mutableProperties =
      createMutableProperties(converter, loc, var, nonDeferredParams);
  fir::MutableBoxValue box(boxAddr, nonDeferredParams, mutableProperties);
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (!var.isGlobal() && !Fortran::semantics::IsDummy(var.getSymbol()))
    fir::factory::disassociateMutableBox(builder, loc, box,
                                         /*polymorphicSetType=*/false);
  return box;
}

//===----------------------------------------------------------------------===//
// MutableBoxValue reading interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::isArraySectionWithoutVectorSubscript(
    const Fortran::lower::SomeExpr &expr) {
  return expr.Rank() > 0 && Fortran::evaluate::IsVariable(expr) &&
         !Fortran::evaluate::UnwrapWholeSymbolDataRef(expr) &&
         !Fortran::evaluate::HasVectorSubscript(expr);
}

void Fortran::lower::associateMutableBox(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const fir::MutableBoxValue &box, const Fortran::lower::SomeExpr &source,
    mlir::ValueRange lbounds, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(source)) {
    fir::factory::disassociateMutableBox(builder, loc, box);
    return;
  }

  // The right hand side is not be evaluated into a temp. Array sections can
  // typically be represented as a value of type `!fir.box`. However, an
  // expression that uses vector subscripts cannot be emboxed. In that case,
  // generate a reference to avoid having to later use a fir.rebox to implement
  // the pointer association.
  fir::ExtendedValue rhs = isArraySectionWithoutVectorSubscript(source)
                               ? converter.genExprBox(loc, source, stmtCtx)
                               : converter.genExprAddr(loc, source, stmtCtx);

  fir::factory::associateMutableBox(builder, loc, box, rhs, lbounds);
}

bool Fortran::lower::isWholeAllocatable(const Fortran::lower::SomeExpr &expr) {
  if (const Fortran::semantics::Symbol *sym =
          Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(expr))
    return Fortran::semantics::IsAllocatable(*sym);
  return false;
}

bool Fortran::lower::isWholePointer(const Fortran::lower::SomeExpr &expr) {
  if (const Fortran::semantics::Symbol *sym =
          Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(expr))
    return Fortran::semantics::IsPointer(*sym);
  return false;
}

mlir::Value Fortran::lower::getAssumedCharAllocatableOrPointerLen(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Value box) {
  // Read length from fir.box (explicit expr cannot safely be re-evaluated
  // here).
  auto readLength = [&]() {
    fir::BoxValue boxLoad =
        builder.create<fir::LoadOp>(loc, fir::getBase(box)).getResult();
    return fir::factory::readCharLen(builder, loc, boxLoad);
  };
  if (Fortran::semantics::IsOptional(sym)) {
    mlir::IndexType idxTy = builder.getIndexType();
    // It is not safe to unconditionally read boxes of optionals in case
    // they are absents. According to 15.5.2.12 3 (9), it is illegal to
    // inquire the length of absent optional, even if non deferred, so
    // it's fine to use undefOp in this case.
    auto isPresent = builder.create<fir::IsPresentOp>(loc, builder.getI1Type(),
                                                      fir::getBase(box));
    mlir::Value len =
        builder.genIfOp(loc, {idxTy}, isPresent, true)
            .genThen(
                [&]() { builder.create<fir::ResultOp>(loc, readLength()); })
            .genElse([&]() {
              auto undef = builder.create<fir::UndefOp>(loc, idxTy);
              builder.create<fir::ResultOp>(loc, undef.getResult());
            })
            .getResults()[0];
    return len;
  }

  return readLength();
}

mlir::Value Fortran::lower::getTypeDescAddr(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Fortran::semantics::DerivedTypeSpec &typeSpec) {
  std::string typeName = Fortran::lower::mangle::mangleName(typeSpec);
  std::string typeDescName = fir::NameUniquer::getTypeDescriptorName(typeName);
  auto typeDescGlobal =
      builder.getModule().lookupSymbol<fir::GlobalOp>(typeDescName);
  if (!typeDescGlobal)
    fir::emitFatalError(loc, "type descriptor not defined");
  return builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(typeDescGlobal.getType()),
      typeDescGlobal.getSymbol());
}
