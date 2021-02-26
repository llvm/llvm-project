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
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Parser/parse-tree.h"
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
// Allocatables runtime call generators
//===----------------------------------------------------------------------===//

using namespace Fortran::runtime;
/// Generate runtime call to set the bounds of an allocatable descriptors.
static void genAllocatableSetBounds(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value boxAddress,
                                    mlir::Value dimIndex,
                                    mlir::Value lowerBound,
                                    mlir::Value upperBound) {
  auto callee = Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableSetBounds)>(
      loc, builder);
  llvm::SmallVector<mlir::Value, 4> args{boxAddress, dimIndex, lowerBound,
                                         upperBound};
  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  builder.create<fir::CallOp>(loc, callee, operands);
}

/// Generate runtime call to set the lengths of a character allocatable
/// descriptor.
static void genAllocatableInitCharRtCall(Fortran::lower::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         const fir::MutableBoxValue &box,
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
  args.push_back(builder.createConvert(loc, inputTypes[0], box.getAddr()));
  args.push_back(builder.createConvert(loc, inputTypes[1], len));
  auto kind = box.getEleTy().cast<fir::CharacterType>().getFKind();
  args.push_back(builder.createIntegerConstant(loc, inputTypes[2], kind));
  auto rank = box.rank();
  args.push_back(builder.createIntegerConstant(loc, inputTypes[3], rank));
  // TODO: coarrays
  auto corank = 0;
  args.push_back(builder.createIntegerConstant(loc, inputTypes[4], corank));
  builder.create<fir::CallOp>(loc, callee, args);
}

/// Generate runtime call to allocate the memory
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
/// Generate runtime call to deallocate the memory
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

//===----------------------------------------------------------------------===//
// MutableBoxValue writer and reader
//===----------------------------------------------------------------------===//

namespace {
/// MutablePropertyWriter and MutablePropertyReader implementations are the only
/// places that depend on how the properties of MutableBoxValue (pointers and
/// allocatables) that can be modified in the lifetime of the entity (address,
/// extents, lower bounds, length parameters) are represented.
/// That is, the properties may be only stored in a fir.box in memory if we
/// need to enforce a single point of truth for the properties across calls.
/// Or, they can be tracked as independent local variables when it is safe to
/// do so. Using bare variables benefits from all optimization passes, even
/// when they are not aware of what a fir.box is and fir.box have not been
/// optimized out yet.

/// MutablePropertyWriter allows reading the properties of a MutableBoxValue.
class MutablePropertyReader {
public:
  MutablePropertyReader(Fortran::lower::FirOpBuilder &builder,
                        mlir::Location loc, const fir::MutableBoxValue &box,
                        bool forceIRBoxRead = false)
      : builder{builder}, loc{loc}, box{box} {
    if (forceIRBoxRead || !box.isDescribedByVariables())
      irBox = builder.create<fir::LoadOp>(loc, box.getAddr());
  }
  /// Get base address of allocated/associated entity.
  mlir::Value readBaseAddress() {
    if (irBox) {
      auto heapOrPtrTy = box.getBoxTy().getEleTy();
      return builder.create<fir::BoxAddrOp>(loc, heapOrPtrTy, irBox);
    }
    auto addrVar = box.getMutableProperties().addr;
    return builder.create<fir::LoadOp>(loc, addrVar);
  }
  /// Return {lbound, extent} values read from the MutableBoxValue given
  /// the dimension.
  std::pair<mlir::Value, mlir::Value> readShape(unsigned dim) {
    auto idxTy = builder.getIndexType();
    if (irBox) {
      auto dimVal = builder.createIntegerConstant(loc, idxTy, dim);
      auto dimInfo = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                    irBox, dimVal);
      return {dimInfo.getResult(0), dimInfo.getResult(1)};
    }
    const auto &mutableProperties = box.getMutableProperties();
    auto lb = builder.create<fir::LoadOp>(loc, mutableProperties.lbounds[dim]);
    auto ext = builder.create<fir::LoadOp>(loc, mutableProperties.extents[dim]);
    return {lb, ext};
  }

  /// Return the character length. If the length was not deferred, the value
  /// that was specified is returned (The mutable fields is not read).
  mlir::Value readCharacterLength() {
    if (box.hasNonDeferredLenParams())
      return box.nonDeferredLenParams()[0];
    if (irBox)
      return Fortran::lower::CharacterExprHelper{builder, loc}
          .readLengthFromBox(irBox);
    const auto &deferred = box.getMutableProperties().deferredParams;
    if (deferred.empty())
      fir::emitFatalError(loc, "allocatable entity has no length property");
    return builder.create<fir::LoadOp>(loc, deferred[0]);
  }
  /// Read all mutable properties. Return the base address.
  mlir::Value read(llvm::SmallVectorImpl<mlir::Value> &lbounds,
                   llvm::SmallVectorImpl<mlir::Value> &extents,
                   llvm::SmallVectorImpl<mlir::Value> &lengths) {
    auto rank = box.rank();
    for (decltype(rank) dim = 0; dim < rank; ++dim) {
      auto [lb, extent] = readShape(dim);
      lbounds.push_back(lb);
      extents.push_back(extent);
    }
    if (box.isCharacter())
      lengths.emplace_back(readCharacterLength());
    else if (box.isDerived())
      mlir::emitError(
          loc, "TODO: read allocatable or pointer derived type LEN parameters");
    return readBaseAddress();
  }

  /// Return the loaded fir.box.
  mlir::Value getIrBox() const {
    assert(irBox);
    return irBox;
  }

  /// Read the lower bounds
  void getLowerBounds(llvm::SmallVectorImpl<mlir::Value> &lbounds) {
    auto rank = box.rank();
    for (decltype(rank) dim = 0; dim < rank; ++dim)
      lbounds.push_back(std::get<0>(readShape(dim)));
  }

private:
  Fortran::lower::FirOpBuilder &builder;
  mlir::Location loc;
  fir::MutableBoxValue box;
  mlir::Value irBox;
};

/// MutablePropertyWriter allows modifying the properties of a MutableBoxValue.
class MutablePropertyWriter {
public:
  MutablePropertyWriter(Fortran::lower::FirOpBuilder &builder,
                        mlir::Location loc, const fir::MutableBoxValue &box)
      : builder{builder}, loc{loc}, box{box} {}
  /// Update MutableBoxValue with new address, shape and length parameters.
  /// Extents and lbounds must all have index type.
  /// lbounds can be empty in which case all ones is assumed.
  /// Length parameters must be provided for the length parameters that are
  /// deferred.
  void updateMutableBox(mlir::Value addr, mlir::ValueRange lbounds,
                        mlir::ValueRange extents, mlir::ValueRange lengths) {
    if (box.isDescribedByVariables())
      updateMutableProperties(addr, lbounds, extents, lengths);
    else
      updateIRBox(addr, lbounds, extents, lengths);
  }
  /// Set unallocated/disassociated status for the entity described by
  /// MutableBoxValue. Deallocation is not performed by this helper.
  void setUnallocatedSatus() {
    if (box.isDescribedByVariables()) {
      auto addrVar = box.getMutableProperties().addr;
      auto nullTy = fir::dyn_cast_ptrEleTy(addrVar.getType());
      builder.create<fir::StoreOp>(loc, builder.createNullConstant(loc, nullTy),
                                   addrVar);
    } else {
      auto deallocatedBox = createUnallocatedBox(builder, loc, box.getBoxTy(),
                                                 box.nonDeferredLenParams());
      builder.create<fir::StoreOp>(loc, deallocatedBox, box.getAddr());
    }
  }

  /// Copy Values from the fir.box into the property variables if any.
  void syncMutablePropertiesFromIRBox() {
    if (!box.isDescribedByVariables())
      return;
    llvm::SmallVector<mlir::Value, 2> lbounds;
    llvm::SmallVector<mlir::Value, 2> extents;
    llvm::SmallVector<mlir::Value, 2> lengths;
    auto addr =
        MutablePropertyReader{builder, loc, box, /*forceIRBoxRead=*/true}.read(
            lbounds, extents, lengths);
    updateMutableProperties(addr, lbounds, extents, lengths);
  }

  /// Copy Values from property variables, if any, into the fir.box.
  void syncIRBoxFromMutableProperties() {
    if (!box.isDescribedByVariables())
      return;
    llvm::SmallVector<mlir::Value, 2> lbounds;
    llvm::SmallVector<mlir::Value, 2> extents;
    llvm::SmallVector<mlir::Value, 2> lengths;
    auto addr = MutablePropertyReader{builder, loc, box}.read(lbounds, extents,
                                                              lengths);
    updateIRBox(addr, lbounds, extents, lengths);
  }

private:
  /// Update the IR box (fir.ref<fir.box<T>>) of the MutableBoxValue.
  void updateIRBox(mlir::Value addr, mlir::ValueRange lbounds,
                   mlir::ValueRange extents, mlir::ValueRange lengths) {
    mlir::Value shape;
    if (!extents.empty()) {
      if (lbounds.empty()) {
        auto shapeType =
            fir::ShapeType::get(builder.getContext(), extents.size());
        shape = builder.create<fir::ShapeOp>(loc, shapeType, extents);
      } else {
        llvm::SmallVector<mlir::Value, 2> shapeShiftBounds;
        for (auto [lb, extent] : llvm::zip(lbounds, extents)) {
          shapeShiftBounds.emplace_back(lb);
          shapeShiftBounds.emplace_back(extent);
        }
        auto shapeShiftType =
            fir::ShapeShiftType::get(builder.getContext(), extents.size());
        shape = builder.create<fir::ShapeShiftOp>(loc, shapeShiftType,
                                                  shapeShiftBounds);
      }
    }
    mlir::Value emptySlice;
    // Ignore lengths if already constant in the box type (this would trigger an
    // error in the embox).
    llvm::SmallVector<mlir::Value, 2> cleanedLengths;
    if (auto charTy = box.getEleTy().dyn_cast<fir::CharacterType>()) {
      if (charTy.getLen() == fir::CharacterType::unknownLen())
        cleanedLengths.append(lengths.begin(), lengths.end());
    } else if (box.isDerived()) {
      // TODO: derived type lengths clean-up
      cleanedLengths = lengths;
    }
    auto irBox = builder.create<fir::EmboxOp>(loc, box.getBoxTy(), addr, shape,
                                              emptySlice, cleanedLengths);
    builder.create<fir::StoreOp>(loc, irBox, box.getAddr());
  }
  /// Update the set of property variables of the MutableBoxValue.
  void updateMutableProperties(mlir::Value addr, mlir::ValueRange lbounds,
                               mlir::ValueRange extents,
                               mlir::ValueRange lengths) {
    const auto &mutableProperties = box.getMutableProperties();
    builder.create<fir::StoreOp>(loc, addr, mutableProperties.addr);
    for (auto [extent, extentVar] :
         llvm::zip(extents, mutableProperties.extents))
      builder.create<fir::StoreOp>(loc, extent, extentVar);
    if (!mutableProperties.lbounds.empty()) {
      if (lbounds.empty()) {
        auto one =
            builder.createIntegerConstant(loc, builder.getIndexType(), 1);
        for (auto lboundVar : mutableProperties.lbounds)
          builder.create<fir::StoreOp>(loc, one, lboundVar);
      } else {
        for (auto [lbound, lboundVar] :
             llvm::zip(lbounds, mutableProperties.lbounds))
          builder.create<fir::StoreOp>(loc, lbound, lboundVar);
      }
    }
    if (box.isCharacter())
      // llvm::zip account for the fact that the length only needs to be stored
      // when it is specified in the allocation and deferred in the
      // MutableBoxValue.
      for (auto [len, lenVar] :
           llvm::zip(lengths, mutableProperties.deferredParams))
        builder.create<fir::StoreOp>(loc, len, lenVar);
    else if (box.isDerived())
      mlir::emitError(
          loc, "TODO: update allocatable derived type length parameters");
  }
  Fortran::lower::FirOpBuilder &builder;
  mlir::Location loc;
  fir::MutableBoxValue box;
};

} // namespace

//===----------------------------------------------------------------------===//
// Allocate statement implementation
//===----------------------------------------------------------------------===//

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
      return converter.genExprMutableBox(loc, *maybeExpr);
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
      TODO(loc, "lower stat expr in allocate and deallocate");
      hasStat = builder.createBool(loc, true);
    } else {
      hasStat = builder.createBool(loc, false);
    }

    if (errMsgExpr)
      TODO(loc, "errmsg in allocate and deallocate");
    else
      errMsgBoxAddr = builder.createNullConstant(loc);
    sourceFile = Fortran::lower::locationToFilename(builder, loc);
    sourceLine = Fortran::lower::locationToLineNo(builder, loc,
                                                  builder.getIntegerType(32));
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
    errorManagement.lower(converter, loc, statExpr, errMsgExpr);
    // Create a landing block after all allocations so that
    // we can jump there in case of error.
    if (errorManagement.hasErrorRecovery())
      TODO(loc, "error recovery");

    // TODO lower source and mold.
    if (sourceExpr || moldExpr)
      TODO(loc, "lower MOLD/SOURCE expr in allocate");

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
    TODO(loc, "Error hanlding in allocate statement");
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
                            const fir::MutableBoxValue &box) {
    llvm::SmallVector<mlir::Value, 2> lbounds;
    llvm::SmallVector<mlir::Value, 2> extents;
    Fortran::lower::StatementContext stmtCtx;
    auto idxTy = builder.getIndexType();
    auto lBoundsAreOnes = lowerBoundsAreOnes(alloc);
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    for (const auto &shapeSpec : alloc.getShapeSpecs()) {
      mlir::Value lb;
      if (!lBoundsAreOnes) {
        if (const auto &lbExpr = std::get<0>(shapeSpec.t)) {
          lb = fir::getBase(converter.genExprValue(
              Fortran::semantics::GetExpr(*lbExpr), stmtCtx, loc));
          lb = builder.createConvert(loc, idxTy, lb);
        } else {
          lb = one;
        }
        lbounds.emplace_back(lb);
      }
      auto ub = fir::getBase(converter.genExprValue(
          Fortran::semantics::GetExpr(std::get<1>(shapeSpec.t)), stmtCtx, loc));
      ub = builder.createConvert(loc, idxTy, ub);
      if (lb) {
        auto diff = builder.create<mlir::SubIOp>(loc, ub, lb);
        extents.emplace_back(builder.create<mlir::AddIOp>(loc, diff, one));
      } else {
        extents.emplace_back(ub);
      }
    }

    llvm::SmallVector<mlir::Value, 2> lengths;
    if (auto charTy = box.getEleTy().dyn_cast<fir::CharacterType>()) {
      if (charTy.getLen() == fir::CharacterType::unknownLen()) {
        if (box.hasNonDeferredLenParams())
          lengths.emplace_back(
              builder.createConvert(loc, idxTy, box.nonDeferredLenParams()[0]));
        else if (!lenParams.empty())
          lengths.emplace_back(builder.createConvert(loc, idxTy, lenParams[0]));
        else
          mlir::emitError(
              loc,
              "could not deduce character lengths in character allocation");
      }
    }

    // FIXME: AllocMemOp is ignoring its length arguments. Squeezed in into the
    // extents for now.
    llvm::SmallVector<mlir::Value, 2> sizes = extents;
    sizes.append(lengths.begin(), lengths.end());
    mlir::Value heap = builder.create<fir::AllocMemOp>(
        loc, box.getBaseTy(), mangleAlloc(alloc), llvm::None, sizes);
    MutablePropertyWriter{builder, loc, box}.updateMutableBox(heap, lbounds,
                                                              extents, lengths);
  }

  void genSimpleAllocation(const Allocation &alloc,
                           const fir::MutableBoxValue &box) {
    if (!box.isDerived() && !errorManagement.hasErrorRecovery() &&
        !alloc.type.IsPolymorphic() && !alloc.hasCoarraySpec() &&
        !useAllocateRuntime) {
      genInlinedAllocation(alloc, box);
      return;
    }
    // Use runtime. sync MutableBoxValue and descriptor before and after calls.
    Fortran::lower::getMutableIRBox(builder, loc, box);
    if (alloc.hasCoarraySpec())
      TODO(loc, "coarray allocation");
    if (alloc.type.IsPolymorphic())
      genSetType(alloc, box);
    genSetDeferredLengthParameters(alloc, box);
    // Set bounds for arrays
    auto idxTy = builder.getIndexType();
    auto i32Ty = builder.getIntegerType(32);
    Fortran::lower::StatementContext stmtCtx;
    auto addr = box.getAddr();
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
    Fortran::lower::syncMutableBoxFromIRBox(builder, loc, box);
    if (auto statAddr = getStatAddr()) {
      auto castStat = builder.createConvert(
          loc, fir::dyn_cast_ptrEleTy(statAddr.getType()), stat);
      builder.create<fir::StoreOp>(loc, castStat, statAddr);
    }
  }

  /// Lower the length parameters that may be specified in the optional
  /// type specification.
  void lowerAllocateLengthParameters() {
    const auto *typeSpec = getIfAllocateStmtTypeSpec();
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
                                      const fir::MutableBoxValue &box) {
    if (lenParams.empty())
      return;
    // TODO: in case a length parameter was not deferred, insert a runtime check
    // that the length is the same (AllocatableCheckLengthParameter runtime
    // call).
    if (box.isCharacter())
      genAllocatableInitCharRtCall(builder, loc, box, lenParams[0]);

    if (box.isDerived())
      TODO(loc, "derived type length parameters in allocate");
  }

  void genSourceAllocation(const Allocation &, const fir::MutableBoxValue &) {
    TODO(loc, "SOURCE allocation lowering");
  }
  void genMoldAllocation(const Allocation &, const fir::MutableBoxValue &) {
    TODO(loc, "MOLD allocation lowering");
  }
  void genSetType(const Allocation &, const fir::MutableBoxValue &) {
    TODO(loc, "Polymorphic entity allocation lowering");
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

  /// Returns a pointer to the DeclTypeSpec if a type-spec is provided in the
  /// allocate statement. Returns a null pointer otherwise.
  const Fortran::semantics::DeclTypeSpec *getIfAllocateStmtTypeSpec() const {
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

//===----------------------------------------------------------------------===//
// Deallocate statement implementation
//===----------------------------------------------------------------------===//

// Generate deallocation of a pointer/allocatable.
static void genDeallocate(Fortran::lower::FirOpBuilder &builder,
                          mlir::Location loc, const fir::MutableBoxValue &box,
                          ErrorManagementValues &errorManagement) {
  // For derived and when error recovery is present, use runtime.
  if (box.isDerived() || errorManagement.hasErrorRecovery() ||
      useAllocateRuntime) {
    // Use runtime with descriptors. Sync MutableBoxValue with its descriptor
    // before and after calls if needed.
    auto irBox = Fortran::lower::getMutableIRBox(builder, loc, box);
    // TODO use return stat for error recovery.
    genAllocatableDeallocate(builder, loc, irBox, errorManagement.hasStat,
                             errorManagement.errMsgBoxAddr,
                             errorManagement.sourceFile,
                             errorManagement.sourceLine);
    Fortran::lower::syncMutableBoxFromIRBox(builder, loc, box);
    return;
  }
  // Inlined deallocate.
  auto addr = MutablePropertyReader(builder, loc, box).readBaseAddress();
  builder.create<fir::FreeMemOp>(loc, addr);
  MutablePropertyWriter{builder, loc, box}.setUnallocatedSatus();
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
    TODO(loc, "error recovery in deallocate");
  ErrorManagementValues errorManagement;
  auto &builder = converter.getFirOpBuilder();
  errorManagement.lower(converter, loc, statExpr, errMsgExpr);
  for (const auto &allocateObject :
       std::get<std::list<Fortran::parser::AllocateObject>>(stmt.t)) {
    auto box = genMutableBoxValue(converter, loc, allocateObject);
    genDeallocate(builder, loc, box, errorManagement);
  }
}

//===----------------------------------------------------------------------===//
// MutableBoxValue creation implementation
//===----------------------------------------------------------------------===//

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
    shape = builder.createShape(
        loc, fir::ArrayBoxValue{nullAddr, extents, /*lbounds=*/llvm::None});
  }
  // Provide dummy length parameters if they are dynamic. If a length parameter
  // is deferred. it is set to zero here and will be set on allocation.
  llvm::SmallVector<mlir::Value, 2> lenParams;
  if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
    if (charTy.getLen() == fir::CharacterType::unknownLen()) {
      if (!nonDeferredParams.empty()) {
        lenParams.push_back(nonDeferredParams[0]);
      } else {
        auto zero = builder.createIntegerConstant(
            loc, builder.getCharacterLengthType(), 0);
        lenParams.push_back(zero);
      }
    }
  }
  mlir::Value emptySlice;
  return builder.create<fir::EmboxOp>(loc, boxType, nullAddr, shape, emptySlice,
                                      lenParams);
}

/// Is this symbol a pointer to a pointer array that does not have the
/// CONTIGUOUS attribute ?
static inline bool
isNonContiguousArrayPointer(const Fortran::semantics::Symbol &sym) {
  return Fortran::semantics::IsPointer(sym) && sym.Rank() != 0 &&
         !sym.attrs().test(Fortran::semantics::Attr::CONTIGUOUS);
}

/// In case it is safe to track the properties in variables outside a
/// descriptor, create the variables to hold the mutable properties of the
/// entity var. The variables are not initialized here.
static fir::MutableProperties
createMutableProperties(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc,
                        const Fortran::lower::pft::Variable &var,
                        mlir::ValueRange nonDeferredParams) {
  auto &builder = converter.getFirOpBuilder();
  const auto &sym = var.getSymbol();
  // Globals and dummies may be associated, creating local variables would
  // require keeping the values and descriptor before and after every single
  // impure calls in the current scope (not only the ones taking the variable as
  // arguments. All.) Volatile means the variable may change in ways not defined
  // per Fortran, so lowering can most likely not keep the descriptor and values
  // in sync as needed.
  // Pointers to non contiguous arrays need to be represented with a fir.box to
  // account for the discontiguity.
  if (var.isGlobal() || Fortran::semantics::IsDummy(sym) ||
      sym.attrs().test(Fortran::semantics::Attr::VOLATILE) ||
      isNonContiguousArrayPointer(sym) || useAllocateRuntime ||
      useDescForMutableBox)
    return {};
  fir::MutableProperties mutableProperties;
  auto name = converter.mangleName(sym);
  auto baseAddrTy = converter.genType(sym);
  if (auto boxType = baseAddrTy.dyn_cast<fir::BoxType>())
    baseAddrTy = boxType.getEleTy();
  // Allocate variable to hold the address and set it will be set to null in
  // setUnallocatedSatus.
  mutableProperties.addr =
      builder.allocateLocal(loc, baseAddrTy, name + ".addr",
                            /*shape=*/llvm::None, /*lenParams=*/llvm::None);
  // Allocate variables to hold lower bounds and extents.
  auto rank = sym.Rank();
  auto idxTy = builder.getIndexType();
  for (decltype(rank) i = 0; i < rank; ++i) {
    auto lboundVar =
        builder.allocateLocal(loc, idxTy, name + ".lb" + std::to_string(i),
                              /*shape=*/llvm::None, /*lenParams=*/llvm::None);
    auto extentVar =
        builder.allocateLocal(loc, idxTy, name + ".ext" + std::to_string(i),
                              /*shape=*/llvm::None, /*lenParams=*/llvm::None);
    mutableProperties.lbounds.emplace_back(lboundVar);
    mutableProperties.extents.emplace_back(extentVar);
  }

  // Allocate variable to hold deferred length parameters.
  auto eleTy = baseAddrTy;
  if (auto newTy = fir::dyn_cast_ptrEleTy(eleTy))
    eleTy = newTy;
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    eleTy = seqTy.getEleTy();
  if (eleTy.isa<fir::RecordType>()) {
    mlir::emitError(loc, "TODO: deferred length type parameters.");
  }
  if (eleTy.isa<fir::CharacterType>() && nonDeferredParams.empty()) {
    auto lenVar = builder.allocateLocal(loc, builder.getCharacterLengthType(),
                                        name + ".len", /*shape=*/llvm::None,
                                        /*lenParams=*/llvm::None);
    mutableProperties.deferredParams.emplace_back(lenVar);
  }
  return mutableProperties;
}

fir::MutableBoxValue Fortran::lower::createMutableBox(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::pft::Variable &var, mlir::Value boxAddr,
    mlir::ValueRange nonDeferredParams) {

  auto mutableProperties =
      createMutableProperties(converter, loc, var, nonDeferredParams);
  auto box =
      fir::MutableBoxValue(boxAddr, nonDeferredParams, mutableProperties);
  auto &builder = converter.getFirOpBuilder();
  if (!var.isGlobal() && !Fortran::semantics::IsDummy(var.getSymbol()))
    MutablePropertyWriter{builder, loc, box}.setUnallocatedSatus();
  return box;
}

fir::MutableBoxValue
Fortran::lower::createTempMutableBox(Fortran::lower::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Type type,
                                     llvm::StringRef name) {
  auto boxType = fir::BoxType::get(fir::HeapType::get(type));
  auto boxAddr = builder.createTemporary(loc, boxType, name);
  auto box =
      fir::MutableBoxValue(boxAddr, /*nonDeferredParams*/ mlir::ValueRange(),
                           /*mutableProperties*/ {});
  MutablePropertyWriter{builder, loc, box}.setUnallocatedSatus();
  return box;
}

//===----------------------------------------------------------------------===//
// MutableBoxValue reading interface implementation
//===----------------------------------------------------------------------===//

/// Helper to decide if a MutableBoxValue must be read to an IrBoxValue or
/// can be read to a reified box value.
static bool readToIrBoxValue(const fir::MutableBoxValue &box) {
  // If this is described by a set of local variables, the value
  // should not be tracked as a fir.box.
  if (box.isDescribedByVariables())
    return false;
  // Polymorphism might be a source of discontiguity, even on allocatables.
  // Track value as fir.box
  if (box.isDerived() || box.isUnlimitedPolymorphic())
    return true;
  // Intrinsic alloctables are contiguous, no need to track the value by
  // fir.box.
  if (box.isAllocatable() || box.rank() == 0)
    return false;
  // Pointer are known to be contiguous at compile time iff they have the
  // CONTIGUOUS attribute.
  return !fir::valueHasFirAttribute(box.getAddr(),
                                    fir::getContiguousAttrName());
}

Fortran::lower::SymbolBox
Fortran::lower::genMutableBoxRead(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  const fir::MutableBoxValue &box) {
  if (box.hasAssumedRank())
    TODO(loc, "Assumed rank allocatables or pointers");
  llvm::SmallVector<mlir::Value, 2> lbounds;
  llvm::SmallVector<mlir::Value, 2> extents;
  llvm::SmallVector<mlir::Value, 2> lengths;
  if (readToIrBoxValue(box)) {
    auto reader = MutablePropertyReader(builder, loc, box);
    reader.getLowerBounds(lbounds);
    return fir::IrBoxValue{reader.getIrBox(), lbounds,
                           box.nonDeferredLenParams()};
  }
  // Contiguous intrinsic type entity: all the data can be extracted from the
  // fir.box.
  auto addr =
      MutablePropertyReader(builder, loc, box).read(lbounds, extents, lengths);
  auto rank = box.rank();
  if (box.isCharacter()) {
    auto len = lengths.empty() ? mlir::Value{} : lengths[0];
    if (rank)
      return fir::CharArrayBoxValue{addr, len, extents, lbounds};
    return fir::CharBoxValue{addr, len};
  }
  if (rank)
    return fir::ArrayBoxValue{addr, extents, lbounds};
  return fir::AbstractBox{addr};
}

mlir::Value Fortran::lower::genIsAllocatedOrAssociatedTest(
    Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
    const fir::MutableBoxValue &box) {
  auto addr = MutablePropertyReader(builder, loc, box).readBaseAddress();
  auto intPtrTy = builder.getIntPtrType();
  auto ptrToInt = builder.createConvert(loc, intPtrTy, addr);
  auto c0 = builder.createIntegerConstant(loc, intPtrTy, 0);
  return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ne, ptrToInt,
                                      c0);
}

//===----------------------------------------------------------------------===//
// MutableBoxValue syncing implementation
//===----------------------------------------------------------------------===//

/// Depending on the implementation, allocatable/pointer descriptor and the
/// MutableBoxValue need to be synced before and after calls passing the
/// descriptor. These calls will generate the syncing if needed and be no-op
mlir::Value
Fortran::lower::getMutableIRBox(Fortran::lower::FirOpBuilder &builder,
                                mlir::Location loc,
                                const fir::MutableBoxValue &box) {
  MutablePropertyWriter{builder, loc, box}.syncIRBoxFromMutableProperties();
  return box.getAddr();
}
void Fortran::lower::syncMutableBoxFromIRBox(
    Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
    const fir::MutableBoxValue &box) {
  MutablePropertyWriter{builder, loc, box}.syncMutablePropertiesFromIRBox();
}
