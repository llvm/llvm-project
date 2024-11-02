//===- SimplifyIntrinsics.cpp -- replace intrinsics with simpler form -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass looks for suitable calls to runtime library for intrinsics that
/// can be simplified/specialized and replaces with a specialized function.
///
/// For example, SUM(arr) can be specialized as a simple function with one loop,
/// compared to the three arguments (plus file & line info) that the runtime
/// call has - when the argument is a 1D-array (multiple loops may be needed
//  for higher dimension arrays, of course)
///
/// The general idea is that besides making the call simpler, it can also be
/// inlined by other passes that run after this pass, which further improves
/// performance, particularly when the work done in the function is trivial
/// and small in size.
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <optional>

namespace fir {
#define GEN_PASS_DEF_SIMPLIFYINTRINSICS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-simplify-intrinsics"

namespace {

class SimplifyIntrinsicsPass
    : public fir::impl::SimplifyIntrinsicsBase<SimplifyIntrinsicsPass> {
  using FunctionTypeGeneratorTy =
      llvm::function_ref<mlir::FunctionType(fir::FirOpBuilder &)>;
  using FunctionBodyGeneratorTy =
      llvm::function_ref<void(fir::FirOpBuilder &, mlir::func::FuncOp &)>;
  using GenReductionBodyTy = llvm::function_ref<void(
      fir::FirOpBuilder &builder, mlir::func::FuncOp &funcOp, unsigned rank,
      mlir::Type elementType)>;

public:
  /// Generate a new function implementing a simplified version
  /// of a Fortran runtime function defined by \p basename name.
  /// \p typeGenerator is a callback that generates the new function's type.
  /// \p bodyGenerator is a callback that generates the new function's body.
  /// The new function is created in the \p builder's Module.
  mlir::func::FuncOp getOrCreateFunction(fir::FirOpBuilder &builder,
                                         const mlir::StringRef &basename,
                                         FunctionTypeGeneratorTy typeGenerator,
                                         FunctionBodyGeneratorTy bodyGenerator);
  void runOnOperation() override;
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

private:
  /// Helper functions to replace a reduction type of call with its
  /// simplified form. The actual function is generated using a callback
  /// function.
  /// \p call is the call to be replaced
  /// \p kindMap is used to create FIROpBuilder
  /// \p genBodyFunc is the callback that builds the replacement function
  void simplifyIntOrFloatReduction(fir::CallOp call,
                                   const fir::KindMapping &kindMap,
                                   GenReductionBodyTy genBodyFunc);
  void simplifyLogicalDim0Reduction(fir::CallOp call,
                                    const fir::KindMapping &kindMap,
                                    GenReductionBodyTy genBodyFunc);
  void simplifyLogicalDim1Reduction(fir::CallOp call,
                                    const fir::KindMapping &kindMap,
                                    GenReductionBodyTy genBodyFunc);
  void simplifyMinMaxlocReduction(fir::CallOp call,
                                  const fir::KindMapping &kindMap, bool isMax);
  void simplifyReductionBody(fir::CallOp call, const fir::KindMapping &kindMap,
                             GenReductionBodyTy genBodyFunc,
                             fir::FirOpBuilder &builder,
                             const mlir::StringRef &basename,
                             mlir::Type elementType);
};

} // namespace

/// Create FirOpBuilder with the provided \p op insertion point
/// and \p kindMap additionally inheriting FastMathFlags from \p op.
static fir::FirOpBuilder
getSimplificationBuilder(mlir::Operation *op, const fir::KindMapping &kindMap) {
  fir::FirOpBuilder builder{op, kindMap};
  auto fmi = mlir::dyn_cast<mlir::arith::ArithFastMathInterface>(*op);
  if (!fmi)
    return builder;

  // Regardless of what default FastMathFlags are used by FirOpBuilder,
  // override them with FastMathFlags attached to the operation.
  builder.setFastMathFlags(fmi.getFastMathFlagsAttr().getValue());
  return builder;
}

/// Generate function type for the simplified version of RTNAME(Sum) and
/// similar functions with a fir.box<none> type returning \p elementType.
static mlir::FunctionType genNoneBoxType(fir::FirOpBuilder &builder,
                                         const mlir::Type &elementType) {
  mlir::Type boxType = fir::BoxType::get(builder.getNoneType());
  return mlir::FunctionType::get(builder.getContext(), {boxType},
                                 {elementType});
}

template <typename Op>
Op expectOp(mlir::Value val) {
  if (Op op = mlir::dyn_cast_or_null<Op>(val.getDefiningOp()))
    return op;
  LLVM_DEBUG(llvm::dbgs() << "Didn't find expected " << Op::getOperationName()
                          << '\n');
  return nullptr;
}

template <typename Op>
static mlir::Value findDefSingle(fir::ConvertOp op) {
  if (auto defOp = expectOp<Op>(op->getOperand(0))) {
    return defOp.getResult();
  }
  return {};
}

template <typename... Ops>
static mlir::Value findDef(fir::ConvertOp op) {
  mlir::Value defOp;
  // Loop over the operation types given to see if any match, exiting once
  // a match is found. Cast to void is needed to avoid compiler complaining
  // that the result of expression is unused
  (void)((defOp = findDefSingle<Ops>(op), (defOp)) || ...);
  return defOp;
}

static bool isOperandAbsent(mlir::Value val) {
  if (auto op = expectOp<fir::ConvertOp>(val)) {
    assert(op->getOperands().size() != 0);
    return mlir::isa_and_nonnull<fir::AbsentOp>(
        op->getOperand(0).getDefiningOp());
  }
  return false;
}

static bool isTrueOrNotConstant(mlir::Value val) {
  if (auto op = expectOp<mlir::arith::ConstantOp>(val)) {
    return !mlir::matchPattern(val, mlir::m_Zero());
  }
  return true;
}

static bool isZero(mlir::Value val) {
  if (auto op = expectOp<fir::ConvertOp>(val)) {
    assert(op->getOperands().size() != 0);
    if (mlir::Operation *defOp = op->getOperand(0).getDefiningOp())
      return mlir::matchPattern(defOp, mlir::m_Zero());
  }
  return false;
}

static mlir::Value findBoxDef(mlir::Value val) {
  if (auto op = expectOp<fir::ConvertOp>(val)) {
    assert(op->getOperands().size() != 0);
    return findDef<fir::EmboxOp, fir::ReboxOp>(op);
  }
  return {};
}

static mlir::Value findMaskDef(mlir::Value val) {
  if (auto op = expectOp<fir::ConvertOp>(val)) {
    assert(op->getOperands().size() != 0);
    return findDef<fir::EmboxOp, fir::ReboxOp, fir::AbsentOp>(op);
  }
  return {};
}

static unsigned getDimCount(mlir::Value val) {
  // In order to find the dimensions count, we look for EmboxOp/ReboxOp
  // and take the count from its *result* type. Note that in case
  // of sliced emboxing the operand and the result of EmboxOp/ReboxOp
  // have different types.
  // Actually, we can take the box type from the operand of
  // the first ConvertOp that has non-opaque box type that we meet
  // going through the ConvertOp chain.
  if (mlir::Value emboxVal = findBoxDef(val))
    if (auto boxTy = emboxVal.getType().dyn_cast<fir::BoxType>())
      if (auto seqTy = boxTy.getEleTy().dyn_cast<fir::SequenceType>())
        return seqTy.getDimension();
  return 0;
}

/// Given the call operation's box argument \p val, discover
/// the element type of the underlying array object.
/// \returns the element type or std::nullopt if the type cannot
/// be reliably found.
/// We expect that the argument is a result of fir.convert
/// with the destination type of !fir.box<none>.
static std::optional<mlir::Type> getArgElementType(mlir::Value val) {
  mlir::Operation *defOp;
  do {
    defOp = val.getDefiningOp();
    // Analyze only sequences of convert operations.
    if (!mlir::isa<fir::ConvertOp>(defOp))
      return std::nullopt;
    val = defOp->getOperand(0);
    // The convert operation is expected to convert from one
    // box type to another box type.
    auto boxType = val.getType().cast<fir::BoxType>();
    auto elementType = fir::unwrapSeqOrBoxedSeqType(boxType);
    if (!elementType.isa<mlir::NoneType>())
      return elementType;
  } while (true);
}

using BodyOpGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &, mlir::Value,
    mlir::Value)>;
using ContinueLoopGenTy = llvm::function_ref<llvm::SmallVector<mlir::Value>(
    fir::FirOpBuilder &, mlir::Location, mlir::Value)>;

/// Generate the reduction loop into \p funcOp.
///
/// \p initVal is a function, called to get the initial value for
///    the reduction value
/// \p genBody is called to fill in the actual reduciton operation
///    for example add for SUM, MAX for MAXVAL, etc.
/// \p rank is the rank of the input argument.
/// \p elementType is the type of the elements in the input array,
///    which may be different to the return type.
/// \p loopCond is called to generate the condition to continue or
///    not for IterWhile loops
/// \p unorderedOrInitalLoopCond contains either a boolean or bool
///    mlir constant, and controls the inital value for while loops
///    or if DoLoop is ordered/unordered.

template <typename OP, typename T, int resultIndex>
static void
genReductionLoop(fir::FirOpBuilder &builder, mlir::func::FuncOp &funcOp,
                 fir::InitValGeneratorTy initVal, ContinueLoopGenTy loopCond,
                 T unorderedOrInitialLoopCond, BodyOpGeneratorTy genBody,
                 unsigned rank, mlir::Type elementType, mlir::Location loc) {

  mlir::IndexType idxTy = builder.getIndexType();

  mlir::Block::BlockArgListType args = funcOp.front().getArguments();
  mlir::Value arg = args[0];

  mlir::Value zeroIdx = builder.createIntegerConstant(loc, idxTy, 0);

  fir::SequenceType::Shape flatShape(rank,
                                     fir::SequenceType::getUnknownExtent());
  mlir::Type arrTy = fir::SequenceType::get(flatShape, elementType);
  mlir::Type boxArrTy = fir::BoxType::get(arrTy);
  mlir::Value array = builder.create<fir::ConvertOp>(loc, boxArrTy, arg);
  mlir::Type resultType = funcOp.getResultTypes()[0];
  mlir::Value init = initVal(builder, loc, resultType);

  llvm::SmallVector<mlir::Value, Fortran::common::maxRank> bounds;

  assert(rank > 0 && "rank cannot be zero");
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);

  // Compute all the upper bounds before the loop nest.
  // It is not strictly necessary for performance, since the loop nest
  // does not have any store operations and any LICM optimization
  // should be able to optimize the redundancy.
  for (unsigned i = 0; i < rank; ++i) {
    mlir::Value dimIdx = builder.createIntegerConstant(loc, idxTy, i);
    auto dims =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, array, dimIdx);
    mlir::Value len = dims.getResult(1);
    // We use C indexing here, so len-1 as loopcount
    mlir::Value loopCount = builder.create<mlir::arith::SubIOp>(loc, len, one);
    bounds.push_back(loopCount);
  }
  // Create a loop nest consisting of OP operations.
  // Collect the loops' induction variables into indices array,
  // which will be used in the innermost loop to load the input
  // array's element.
  // The loops are generated such that the innermost loop processes
  // the 0 dimension.
  llvm::SmallVector<mlir::Value, Fortran::common::maxRank> indices;
  for (unsigned i = rank; 0 < i; --i) {
    mlir::Value step = one;
    mlir::Value loopCount = bounds[i - 1];
    auto loop = builder.create<OP>(loc, zeroIdx, loopCount, step,
                                   unorderedOrInitialLoopCond,
                                   /*finalCountValue=*/false, init);
    init = loop.getRegionIterArgs()[resultIndex];
    indices.push_back(loop.getInductionVar());
    // Set insertion point to the loop body so that the next loop
    // is inserted inside the current one.
    builder.setInsertionPointToStart(loop.getBody());
  }

  // Reverse the indices such that they are ordered as:
  //   <dim-0-idx, dim-1-idx, ...>
  std::reverse(indices.begin(), indices.end());
  // We are in the innermost loop: generate the reduction body.
  mlir::Type eleRefTy = builder.getRefType(elementType);
  mlir::Value addr =
      builder.create<fir::CoordinateOp>(loc, eleRefTy, array, indices);
  mlir::Value elem = builder.create<fir::LoadOp>(loc, addr);
  mlir::Value reductionVal = genBody(builder, loc, elementType, elem, init);
  // Generate vector with condition to continue while loop at [0] and result
  // from current loop at [1] for IterWhileOp loops, just result at [0] for
  // DoLoopOp loops.
  llvm::SmallVector<mlir::Value> results = loopCond(builder, loc, reductionVal);

  // Unwind the loop nest and insert ResultOp on each level
  // to return the updated value of the reduction to the enclosing
  // loops.
  for (unsigned i = 0; i < rank; ++i) {
    auto result = builder.create<fir::ResultOp>(loc, results);
    // Proceed to the outer loop.
    auto loop = mlir::cast<OP>(result->getParentOp());
    results = loop.getResults();
    // Set insertion point after the loop operation that we have
    // just processed.
    builder.setInsertionPointAfter(loop.getOperation());
  }
  // End of loop nest. The insertion point is after the outermost loop.
  // Return the reduction value from the function.
  builder.create<mlir::func::ReturnOp>(loc, results[resultIndex]);
}

static llvm::SmallVector<mlir::Value> nopLoopCond(fir::FirOpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value reductionVal) {
  return {reductionVal};
}

/// Generate function body of the simplified version of RTNAME(Sum)
/// with signature provided by \p funcOp. The caller is responsible
/// for saving/restoring the original insertion point of \p builder.
/// \p funcOp is expected to be empty on entry to this function.
/// \p rank specifies the rank of the input argument.
static void genRuntimeSumBody(fir::FirOpBuilder &builder,
                              mlir::func::FuncOp &funcOp, unsigned rank,
                              mlir::Type elementType) {
  // function RTNAME(Sum)<T>x<rank>_simplified(arr)
  //   T, dimension(:) :: arr
  //   T sum = 0
  //   integer iter
  //   do iter = 0, extent(arr)
  //     sum = sum + arr[iter]
  //   end do
  //   RTNAME(Sum)<T>x<rank>_simplified = sum
  // end function RTNAME(Sum)<T>x<rank>_simplified
  auto zero = [](fir::FirOpBuilder builder, mlir::Location loc,
                 mlir::Type elementType) {
    if (auto ty = elementType.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(loc, elementType,
                                        llvm::APFloat::getZero(sem));
    }
    return builder.createIntegerConstant(loc, elementType, 0);
  };

  auto genBodyOp = [](fir::FirOpBuilder builder, mlir::Location loc,
                      mlir::Type elementType, mlir::Value elem1,
                      mlir::Value elem2) -> mlir::Value {
    if (elementType.isa<mlir::FloatType>())
      return builder.create<mlir::arith::AddFOp>(loc, elem1, elem2);
    if (elementType.isa<mlir::IntegerType>())
      return builder.create<mlir::arith::AddIOp>(loc, elem1, elem2);

    llvm_unreachable("unsupported type");
    return {};
  };

  mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());

  genReductionLoop<fir::DoLoopOp, bool, 0>(builder, funcOp, zero, nopLoopCond,
                                           false, genBodyOp, rank, elementType,
                                           loc);
}

static void genRuntimeMaxvalBody(fir::FirOpBuilder &builder,
                                 mlir::func::FuncOp &funcOp, unsigned rank,
                                 mlir::Type elementType) {
  auto init = [](fir::FirOpBuilder builder, mlir::Location loc,
                 mlir::Type elementType) {
    if (auto ty = elementType.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, elementType, llvm::APFloat::getLargest(sem, /*Negative=*/true));
    }
    unsigned bits = elementType.getIntOrFloatBitWidth();
    int64_t minInt = llvm::APInt::getSignedMinValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, elementType, minInt);
  };

  auto genBodyOp = [](fir::FirOpBuilder builder, mlir::Location loc,
                      mlir::Type elementType, mlir::Value elem1,
                      mlir::Value elem2) -> mlir::Value {
    if (elementType.isa<mlir::FloatType>()) {
      // arith.maxf later converted to llvm.intr.maxnum does not work
      // correctly for NaNs and -0.0 (see maxnum/minnum pattern matching
      // in LLVM's InstCombine pass). Moreover, llvm.intr.maxnum
      // for F128 operands is lowered into fmaxl call by LLVM.
      // This libm function may not work properly for F128 arguments
      // on targets where long double is not F128. It is an LLVM issue,
      // but we just use normal select here to resolve all the cases.
      auto compare = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OGT, elem1, elem2);
      return builder.create<mlir::arith::SelectOp>(loc, compare, elem1, elem2);
    }
    if (elementType.isa<mlir::IntegerType>())
      return builder.create<mlir::arith::MaxSIOp>(loc, elem1, elem2);

    llvm_unreachable("unsupported type");
    return {};
  };

  mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());

  genReductionLoop<fir::DoLoopOp, bool, 0>(builder, funcOp, init, nopLoopCond,
                                           false, genBodyOp, rank, elementType,
                                           loc);
}

static void genRuntimeCountBody(fir::FirOpBuilder &builder,
                                mlir::func::FuncOp &funcOp, unsigned rank,
                                mlir::Type elementType) {
  auto zero = [](fir::FirOpBuilder builder, mlir::Location loc,
                 mlir::Type elementType) {
    unsigned bits = elementType.getIntOrFloatBitWidth();
    int64_t zeroInt = llvm::APInt::getZero(bits).getSExtValue();
    return builder.createIntegerConstant(loc, elementType, zeroInt);
  };

  auto genBodyOp = [](fir::FirOpBuilder builder, mlir::Location loc,
                      mlir::Type elementType, mlir::Value elem1,
                      mlir::Value elem2) -> mlir::Value {
    auto zero32 = builder.createIntegerConstant(loc, elementType, 0);
    auto zero64 = builder.createIntegerConstant(loc, builder.getI64Type(), 0);
    auto one64 = builder.createIntegerConstant(loc, builder.getI64Type(), 1);

    auto compare = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, elem1, zero32);
    auto select =
        builder.create<mlir::arith::SelectOp>(loc, compare, zero64, one64);
    return builder.create<mlir::arith::AddIOp>(loc, select, elem2);
  };

  // Count always gets I32 for elementType as it converts logical input to
  // logical<4> before passing to the function.
  mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());

  genReductionLoop<fir::DoLoopOp, bool, 0>(builder, funcOp, zero, nopLoopCond,
                                           false, genBodyOp, rank, elementType,
                                           loc);
}

static void genRuntimeAnyBody(fir::FirOpBuilder &builder,
                              mlir::func::FuncOp &funcOp, unsigned rank,
                              mlir::Type elementType) {
  auto zero = [](fir::FirOpBuilder builder, mlir::Location loc,
                 mlir::Type elementType) {
    return builder.createIntegerConstant(loc, elementType, 0);
  };

  auto genBodyOp = [](fir::FirOpBuilder builder, mlir::Location loc,
                      mlir::Type elementType, mlir::Value elem1,
                      mlir::Value elem2) -> mlir::Value {
    auto zero = builder.createIntegerConstant(loc, elementType, 0);
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, elem1, zero);
  };

  auto continueCond = [](fir::FirOpBuilder builder, mlir::Location loc,
                         mlir::Value reductionVal) {
    auto one1 = builder.createIntegerConstant(loc, builder.getI1Type(), 1);
    auto eor = builder.create<mlir::arith::XOrIOp>(loc, reductionVal, one1);
    llvm::SmallVector<mlir::Value> results = {eor, reductionVal};
    return results;
  };

  mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());
  mlir::Value ok = builder.createBool(loc, true);

  genReductionLoop<fir::IterWhileOp, mlir::Value, 1>(
      builder, funcOp, zero, continueCond, ok, genBodyOp, rank, elementType,
      loc);
}

static void genRuntimeAllBody(fir::FirOpBuilder &builder,
                              mlir::func::FuncOp &funcOp, unsigned rank,
                              mlir::Type elementType) {
  auto one = [](fir::FirOpBuilder builder, mlir::Location loc,
                mlir::Type elementType) {
    return builder.createIntegerConstant(loc, elementType, 1);
  };

  auto genBodyOp = [](fir::FirOpBuilder builder, mlir::Location loc,
                      mlir::Type elementType, mlir::Value elem1,
                      mlir::Value elem2) -> mlir::Value {
    auto zero = builder.createIntegerConstant(loc, elementType, 0);
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, elem1, zero);
  };

  auto continueCond = [](fir::FirOpBuilder builder, mlir::Location loc,
                         mlir::Value reductionVal) {
    llvm::SmallVector<mlir::Value> results = {reductionVal, reductionVal};
    return results;
  };

  mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());
  mlir::Value ok = builder.createBool(loc, true);

  genReductionLoop<fir::IterWhileOp, mlir::Value, 1>(
      builder, funcOp, one, continueCond, ok, genBodyOp, rank, elementType,
      loc);
}

static mlir::FunctionType genRuntimeMinlocType(fir::FirOpBuilder &builder,
                                               unsigned int rank) {
  mlir::Type boxType = fir::BoxType::get(builder.getNoneType());
  mlir::Type boxRefType = builder.getRefType(boxType);

  return mlir::FunctionType::get(builder.getContext(),
                                 {boxRefType, boxType, boxType}, {});
}

static void genRuntimeMinMaxlocBody(fir::FirOpBuilder &builder,
                                    mlir::func::FuncOp &funcOp, bool isMax,
                                    unsigned rank, int maskRank,
                                    mlir::Type elementType,
                                    mlir::Type maskElemType,
                                    mlir::Type resultElemTy) {
  auto init = [isMax](fir::FirOpBuilder builder, mlir::Location loc,
                      mlir::Type elementType) {
    if (auto ty = elementType.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, elementType, llvm::APFloat::getLargest(sem, /*Negative=*/isMax));
    }
    unsigned bits = elementType.getIntOrFloatBitWidth();
    int64_t initValue = (isMax ? llvm::APInt::getSignedMinValue(bits)
                               : llvm::APInt::getSignedMaxValue(bits))
                            .getSExtValue();
    return builder.createIntegerConstant(loc, elementType, initValue);
  };

  mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());

  mlir::Value mask = funcOp.front().getArgument(2);

  // Set up result array in case of early exit / 0 length array
  mlir::IndexType idxTy = builder.getIndexType();
  mlir::Type resultTy = fir::SequenceType::get(rank, resultElemTy);
  mlir::Type resultHeapTy = fir::HeapType::get(resultTy);
  mlir::Type resultBoxTy = fir::BoxType::get(resultHeapTy);

  mlir::Value returnValue = builder.createIntegerConstant(loc, resultElemTy, 0);
  mlir::Value resultArrSize = builder.createIntegerConstant(loc, idxTy, rank);

  mlir::Value resultArrInit = builder.create<fir::AllocMemOp>(loc, resultTy);
  mlir::Value resultArrShape = builder.create<fir::ShapeOp>(loc, resultArrSize);
  mlir::Value resultArr = builder.create<fir::EmboxOp>(
      loc, resultBoxTy, resultArrInit, resultArrShape);

  mlir::Type resultRefTy = builder.getRefType(resultElemTy);

  if (maskRank > 0) {
    fir::SequenceType::Shape flatShape(rank,
                                       fir::SequenceType::getUnknownExtent());
    mlir::Type maskTy = fir::SequenceType::get(flatShape, maskElemType);
    mlir::Type boxMaskTy = fir::BoxType::get(maskTy);
    mask = builder.create<fir::ConvertOp>(loc, boxMaskTy, mask);
  }

  for (unsigned int i = 0; i < rank; ++i) {
    mlir::Value index = builder.createIntegerConstant(loc, idxTy, i);
    mlir::Value resultElemAddr =
        builder.create<fir::CoordinateOp>(loc, resultRefTy, resultArr, index);
    builder.create<fir::StoreOp>(loc, returnValue, resultElemAddr);
  }

  auto genBodyOp =
      [&rank, &resultArr, isMax, &mask, &maskElemType, &maskRank](
          fir::FirOpBuilder builder, mlir::Location loc, mlir::Type elementType,
          mlir::Value array, mlir::Value flagRef, mlir::Value reduction,
          const llvm::SmallVectorImpl<mlir::Value> &indices) -> mlir::Value {
    // We are in the innermost loop: generate the reduction body.
    if (maskRank > 0) {
      mlir::Type logicalRef = builder.getRefType(maskElemType);
      mlir::Value maskAddr =
          builder.create<fir::CoordinateOp>(loc, logicalRef, mask, indices);
      mlir::Value maskElem = builder.create<fir::LoadOp>(loc, maskAddr);

      // fir::IfOp requires argument to be I1 - won't accept logical or any
      // other Integer.
      mlir::Type ifCompatType = builder.getI1Type();
      mlir::Value ifCompatElem =
          builder.create<fir::ConvertOp>(loc, ifCompatType, maskElem);

      llvm::SmallVector<mlir::Type> resultsTy = {elementType, elementType};
      fir::IfOp ifOp = builder.create<fir::IfOp>(loc, elementType, ifCompatElem,
                                                 /*withElseRegion=*/true);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    // Set flag that mask was true at some point
    mlir::Value flagSet = builder.createIntegerConstant(
        loc, mlir::cast<fir::ReferenceType>(flagRef.getType()).getEleTy(), 1);
    builder.create<fir::StoreOp>(loc, flagSet, flagRef);
    mlir::Type eleRefTy = builder.getRefType(elementType);
    mlir::Value addr =
        builder.create<fir::CoordinateOp>(loc, eleRefTy, array, indices);
    mlir::Value elem = builder.create<fir::LoadOp>(loc, addr);

    mlir::Value cmp;
    if (elementType.isa<mlir::FloatType>()) {
      cmp = builder.create<mlir::arith::CmpFOp>(
          loc,
          isMax ? mlir::arith::CmpFPredicate::OGT
                : mlir::arith::CmpFPredicate::OLT,
          elem, reduction);
    } else if (elementType.isa<mlir::IntegerType>()) {
      cmp = builder.create<mlir::arith::CmpIOp>(
          loc,
          isMax ? mlir::arith::CmpIPredicate::sgt
                : mlir::arith::CmpIPredicate::slt,
          elem, reduction);
    } else {
      llvm_unreachable("unsupported type");
    }

    fir::IfOp ifOp = builder.create<fir::IfOp>(loc, elementType, cmp,
                                               /*withElseRegion*/ true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Type resultElemTy = hlfir::getFortranElementType(resultArr.getType());
    mlir::Type returnRefTy = builder.getRefType(resultElemTy);
    mlir::IndexType idxTy = builder.getIndexType();

    mlir::Value one = builder.createIntegerConstant(loc, resultElemTy, 1);

    for (unsigned int i = 0; i < rank; ++i) {
      mlir::Value index = builder.createIntegerConstant(loc, idxTy, i);
      mlir::Value resultElemAddr =
          builder.create<fir::CoordinateOp>(loc, returnRefTy, resultArr, index);
      mlir::Value convert =
          builder.create<fir::ConvertOp>(loc, resultElemTy, indices[i]);
      mlir::Value fortranIndex =
          builder.create<mlir::arith::AddIOp>(loc, convert, one);
      builder.create<fir::StoreOp>(loc, fortranIndex, resultElemAddr);
    }
    builder.create<fir::ResultOp>(loc, elem);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    builder.create<fir::ResultOp>(loc, reduction);
    builder.setInsertionPointAfter(ifOp);
    mlir::Value reductionVal = ifOp.getResult(0);

    // Close the mask if needed
    if (maskRank > 0) {
      fir::IfOp ifOp =
          mlir::dyn_cast<fir::IfOp>(builder.getBlock()->getParentOp());
      builder.create<fir::ResultOp>(loc, reductionVal);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      builder.create<fir::ResultOp>(loc, reduction);
      reductionVal = ifOp.getResult(0);
      builder.setInsertionPointAfter(ifOp);
    }

    return reductionVal;
  };

  // if mask is a logical scalar, we can check its value before the main loop
  // and either ignore the fact it is there or exit early.
  if (maskRank == 0) {
    mlir::Type logical = builder.getI1Type();
    mlir::IndexType idxTy = builder.getIndexType();

    fir::SequenceType::Shape singleElement(1, 1);
    mlir::Type arrTy = fir::SequenceType::get(singleElement, logical);
    mlir::Type boxArrTy = fir::BoxType::get(arrTy);
    mlir::Value array = builder.create<fir::ConvertOp>(loc, boxArrTy, mask);

    mlir::Value indx = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Type logicalRefTy = builder.getRefType(logical);
    mlir::Value condAddr =
        builder.create<fir::CoordinateOp>(loc, logicalRefTy, array, indx);
    mlir::Value cond = builder.create<fir::LoadOp>(loc, condAddr);

    fir::IfOp ifOp = builder.create<fir::IfOp>(loc, elementType, cond,
                                               /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::Value basicValue;
    if (elementType.isa<mlir::IntegerType>()) {
      basicValue = builder.createIntegerConstant(loc, elementType, 0);
    } else {
      basicValue = builder.createRealConstant(loc, elementType, 0);
    }
    builder.create<fir::ResultOp>(loc, basicValue);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }
  auto getAddrFn = [](fir::FirOpBuilder builder, mlir::Location loc,
                      const mlir::Type &resultElemType, mlir::Value resultArr,
                      mlir::Value index) {
    mlir::Type resultRefTy = builder.getRefType(resultElemType);
    return builder.create<fir::CoordinateOp>(loc, resultRefTy, resultArr,
                                             index);
  };

  genMinMaxlocReductionLoop(builder, funcOp.front().getArgument(1), init,
                            genBodyOp, getAddrFn, rank, elementType, loc,
                            maskElemType, resultArr, maskRank == 0);

  // Store newly created output array to the reference passed in
  fir::SequenceType::Shape resultShape(1, rank);
  mlir::Type outputArrTy = fir::SequenceType::get(resultShape, resultElemTy);
  mlir::Type outputHeapTy = fir::HeapType::get(outputArrTy);
  mlir::Type outputBoxTy = fir::BoxType::get(outputHeapTy);
  mlir::Type outputRefTy = builder.getRefType(outputBoxTy);
  mlir::Value outputArr = builder.create<fir::ConvertOp>(
      loc, outputRefTy, funcOp.front().getArgument(0));

  // Store nearly created array to output array
  builder.create<fir::StoreOp>(loc, resultArr, outputArr);
  builder.create<mlir::func::ReturnOp>(loc);
}

/// Generate function type for the simplified version of RTNAME(DotProduct)
/// operating on the given \p elementType.
static mlir::FunctionType genRuntimeDotType(fir::FirOpBuilder &builder,
                                            const mlir::Type &elementType) {
  mlir::Type boxType = fir::BoxType::get(builder.getNoneType());
  return mlir::FunctionType::get(builder.getContext(), {boxType, boxType},
                                 {elementType});
}

/// Generate function body of the simplified version of RTNAME(DotProduct)
/// with signature provided by \p funcOp. The caller is responsible
/// for saving/restoring the original insertion point of \p builder.
/// \p funcOp is expected to be empty on entry to this function.
/// \p arg1ElementTy and \p arg2ElementTy specify elements types
/// of the underlying array objects - they are used to generate proper
/// element accesses.
static void genRuntimeDotBody(fir::FirOpBuilder &builder,
                              mlir::func::FuncOp &funcOp,
                              mlir::Type arg1ElementTy,
                              mlir::Type arg2ElementTy) {
  // function RTNAME(DotProduct)<T>_simplified(arr1, arr2)
  //   T, dimension(:) :: arr1, arr2
  //   T product = 0
  //   integer iter
  //   do iter = 0, extent(arr1)
  //     product = product + arr1[iter] * arr2[iter]
  //   end do
  //   RTNAME(ADotProduct)<T>_simplified = product
  // end function RTNAME(DotProduct)<T>_simplified
  auto loc = mlir::UnknownLoc::get(builder.getContext());
  mlir::Type resultElementType = funcOp.getResultTypes()[0];
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());

  mlir::IndexType idxTy = builder.getIndexType();

  mlir::Value zero =
      resultElementType.isa<mlir::FloatType>()
          ? builder.createRealConstant(loc, resultElementType, 0.0)
          : builder.createIntegerConstant(loc, resultElementType, 0);

  mlir::Block::BlockArgListType args = funcOp.front().getArguments();
  mlir::Value arg1 = args[0];
  mlir::Value arg2 = args[1];

  mlir::Value zeroIdx = builder.createIntegerConstant(loc, idxTy, 0);

  fir::SequenceType::Shape flatShape = {fir::SequenceType::getUnknownExtent()};
  mlir::Type arrTy1 = fir::SequenceType::get(flatShape, arg1ElementTy);
  mlir::Type boxArrTy1 = fir::BoxType::get(arrTy1);
  mlir::Value array1 = builder.create<fir::ConvertOp>(loc, boxArrTy1, arg1);
  mlir::Type arrTy2 = fir::SequenceType::get(flatShape, arg2ElementTy);
  mlir::Type boxArrTy2 = fir::BoxType::get(arrTy2);
  mlir::Value array2 = builder.create<fir::ConvertOp>(loc, boxArrTy2, arg2);
  // This version takes the loop trip count from the first argument.
  // If the first argument's box has unknown (at compilation time)
  // extent, then it may be better to take the extent from the second
  // argument - so that after inlining the loop may be better optimized, e.g.
  // fully unrolled. This requires generating two versions of the simplified
  // function and some analysis at the call site to choose which version
  // is more profitable to call.
  // Note that we can assume that both arguments have the same extent.
  auto dims =
      builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, array1, zeroIdx);
  mlir::Value len = dims.getResult(1);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  mlir::Value step = one;

  // We use C indexing here, so len-1 as loopcount
  mlir::Value loopCount = builder.create<mlir::arith::SubIOp>(loc, len, one);
  auto loop = builder.create<fir::DoLoopOp>(loc, zeroIdx, loopCount, step,
                                            /*unordered=*/false,
                                            /*finalCountValue=*/false, zero);
  mlir::Value sumVal = loop.getRegionIterArgs()[0];

  // Begin loop code
  mlir::OpBuilder::InsertPoint loopEndPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(loop.getBody());

  mlir::Type eleRef1Ty = builder.getRefType(arg1ElementTy);
  mlir::Value index = loop.getInductionVar();
  mlir::Value addr1 =
      builder.create<fir::CoordinateOp>(loc, eleRef1Ty, array1, index);
  mlir::Value elem1 = builder.create<fir::LoadOp>(loc, addr1);
  // Convert to the result type.
  elem1 = builder.create<fir::ConvertOp>(loc, resultElementType, elem1);

  mlir::Type eleRef2Ty = builder.getRefType(arg2ElementTy);
  mlir::Value addr2 =
      builder.create<fir::CoordinateOp>(loc, eleRef2Ty, array2, index);
  mlir::Value elem2 = builder.create<fir::LoadOp>(loc, addr2);
  // Convert to the result type.
  elem2 = builder.create<fir::ConvertOp>(loc, resultElementType, elem2);

  if (resultElementType.isa<mlir::FloatType>())
    sumVal = builder.create<mlir::arith::AddFOp>(
        loc, builder.create<mlir::arith::MulFOp>(loc, elem1, elem2), sumVal);
  else if (resultElementType.isa<mlir::IntegerType>())
    sumVal = builder.create<mlir::arith::AddIOp>(
        loc, builder.create<mlir::arith::MulIOp>(loc, elem1, elem2), sumVal);
  else
    llvm_unreachable("unsupported type");

  builder.create<fir::ResultOp>(loc, sumVal);
  // End of loop.
  builder.restoreInsertionPoint(loopEndPt);

  mlir::Value resultVal = loop.getResult(0);
  builder.create<mlir::func::ReturnOp>(loc, resultVal);
}

mlir::func::FuncOp SimplifyIntrinsicsPass::getOrCreateFunction(
    fir::FirOpBuilder &builder, const mlir::StringRef &baseName,
    FunctionTypeGeneratorTy typeGenerator,
    FunctionBodyGeneratorTy bodyGenerator) {
  // WARNING: if the function generated here changes its signature
  //          or behavior (the body code), we should probably embed some
  //          versioning information into its name, otherwise libraries
  //          statically linked with older versions of Flang may stop
  //          working with object files created with newer Flang.
  //          We can also avoid this by using internal linkage, but
  //          this may increase the size of final executable/shared library.
  std::string replacementName = mlir::Twine{baseName, "_simplified"}.str();
  mlir::ModuleOp module = builder.getModule();
  // If we already have a function, just return it.
  mlir::func::FuncOp newFunc =
      fir::FirOpBuilder::getNamedFunction(module, replacementName);
  mlir::FunctionType fType = typeGenerator(builder);
  if (newFunc) {
    assert(newFunc.getFunctionType() == fType &&
           "type mismatch for simplified function");
    return newFunc;
  }

  // Need to build the function!
  auto loc = mlir::UnknownLoc::get(builder.getContext());
  newFunc =
      fir::FirOpBuilder::createFunction(loc, module, replacementName, fType);
  auto inlineLinkage = mlir::LLVM::linkage::Linkage::LinkonceODR;
  auto linkage =
      mlir::LLVM::LinkageAttr::get(builder.getContext(), inlineLinkage);
  newFunc->setAttr("llvm.linkage", linkage);

  // Save the position of the original call.
  mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();

  bodyGenerator(builder, newFunc);

  // Now back to where we were adding code earlier...
  builder.restoreInsertionPoint(insertPt);

  return newFunc;
}

void SimplifyIntrinsicsPass::simplifyIntOrFloatReduction(
    fir::CallOp call, const fir::KindMapping &kindMap,
    GenReductionBodyTy genBodyFunc) {
  // args[1] and args[2] are source filename and line number, ignored.
  mlir::Operation::operand_range args = call.getArgs();

  const mlir::Value &dim = args[3];
  const mlir::Value &mask = args[4];
  // dim is zero when it is absent, which is an implementation
  // detail in the runtime library.

  bool dimAndMaskAbsent = isZero(dim) && isOperandAbsent(mask);
  unsigned rank = getDimCount(args[0]);

  // Rank is set to 0 for assumed shape arrays, don't simplify
  // in these cases
  if (!(dimAndMaskAbsent && rank > 0))
    return;

  mlir::Type resultType = call.getResult(0).getType();

  if (!resultType.isa<mlir::FloatType>() &&
      !resultType.isa<mlir::IntegerType>())
    return;

  auto argType = getArgElementType(args[0]);
  if (!argType)
    return;
  assert(*argType == resultType &&
         "Argument/result types mismatch in reduction");

  mlir::SymbolRefAttr callee = call.getCalleeAttr();

  fir::FirOpBuilder builder{getSimplificationBuilder(call, kindMap)};
  std::string fmfString{builder.getFastMathFlagsString()};
  std::string funcName =
      (mlir::Twine{callee.getLeafReference().getValue(), "x"} +
       mlir::Twine{rank} +
       // We must mangle the generated function name with FastMathFlags
       // value.
       (fmfString.empty() ? mlir::Twine{} : mlir::Twine{"_", fmfString}))
          .str();

  simplifyReductionBody(call, kindMap, genBodyFunc, builder, funcName,
                        resultType);
}

void SimplifyIntrinsicsPass::simplifyLogicalDim0Reduction(
    fir::CallOp call, const fir::KindMapping &kindMap,
    GenReductionBodyTy genBodyFunc) {

  mlir::Operation::operand_range args = call.getArgs();
  const mlir::Value &dim = args[3];
  unsigned rank = getDimCount(args[0]);

  // getDimCount returns a rank of 0 for assumed shape arrays, don't simplify in
  // these cases.
  if (!(isZero(dim) && rank > 0))
    return;

  mlir::Value inputBox = findBoxDef(args[0]);

  mlir::Type elementType = hlfir::getFortranElementType(inputBox.getType());
  mlir::SymbolRefAttr callee = call.getCalleeAttr();

  fir::FirOpBuilder builder{getSimplificationBuilder(call, kindMap)};

  // Treating logicals as integers makes things a lot easier
  fir::LogicalType logicalType = {elementType.dyn_cast<fir::LogicalType>()};
  fir::KindTy kind = logicalType.getFKind();
  mlir::Type intElementType = builder.getIntegerType(kind * 8);

  // Mangle kind into function name as it is not done by default
  std::string funcName =
      (mlir::Twine{callee.getLeafReference().getValue(), "Logical"} +
       mlir::Twine{kind} + "x" + mlir::Twine{rank})
          .str();

  simplifyReductionBody(call, kindMap, genBodyFunc, builder, funcName,
                        intElementType);
}

void SimplifyIntrinsicsPass::simplifyLogicalDim1Reduction(
    fir::CallOp call, const fir::KindMapping &kindMap,
    GenReductionBodyTy genBodyFunc) {

  mlir::Operation::operand_range args = call.getArgs();
  mlir::SymbolRefAttr callee = call.getCalleeAttr();
  mlir::StringRef funcNameBase = callee.getLeafReference().getValue();
  unsigned rank = getDimCount(args[0]);

  // getDimCount returns a rank of 0 for assumed shape arrays, don't simplify in
  // these cases. We check for Dim at the end as some logical functions (Any,
  // All) set dim to 1 instead of 0 when the argument is not present.
  if (funcNameBase.ends_with("Dim") || !(rank > 0))
    return;

  mlir::Value inputBox = findBoxDef(args[0]);
  mlir::Type elementType = hlfir::getFortranElementType(inputBox.getType());

  fir::FirOpBuilder builder{getSimplificationBuilder(call, kindMap)};

  // Treating logicals as integers makes things a lot easier
  fir::LogicalType logicalType = {elementType.dyn_cast<fir::LogicalType>()};
  fir::KindTy kind = logicalType.getFKind();
  mlir::Type intElementType = builder.getIntegerType(kind * 8);

  // Mangle kind into function name as it is not done by default
  std::string funcName =
      (mlir::Twine{callee.getLeafReference().getValue(), "Logical"} +
       mlir::Twine{kind} + "x" + mlir::Twine{rank})
          .str();

  simplifyReductionBody(call, kindMap, genBodyFunc, builder, funcName,
                        intElementType);
}

void SimplifyIntrinsicsPass::simplifyMinMaxlocReduction(
    fir::CallOp call, const fir::KindMapping &kindMap, bool isMax) {

  mlir::Operation::operand_range args = call.getArgs();

  mlir::Value back = args[6];
  if (isTrueOrNotConstant(back))
    return;

  mlir::Value mask = args[5];
  mlir::Value maskDef = findMaskDef(mask);

  // maskDef is set to NULL when the defining op is not one we accept.
  // This tends to be because it is a selectOp, in which case let the
  // runtime deal with it.
  if (maskDef == NULL)
    return;

  mlir::SymbolRefAttr callee = call.getCalleeAttr();
  mlir::StringRef funcNameBase = callee.getLeafReference().getValue();
  unsigned rank = getDimCount(args[1]);
  if (funcNameBase.ends_with("Dim") || !(rank > 0))
    return;

  fir::FirOpBuilder builder{getSimplificationBuilder(call, kindMap)};
  mlir::Location loc = call.getLoc();
  auto inputBox = findBoxDef(args[1]);
  mlir::Type inputType = hlfir::getFortranElementType(inputBox.getType());

  if (inputType.isa<fir::CharacterType>())
    return;

  int maskRank;
  fir::KindTy kind = 0;
  mlir::Type logicalElemType = builder.getI1Type();
  if (isOperandAbsent(mask)) {
    maskRank = -1;
  } else {
    maskRank = getDimCount(mask);
    mlir::Type maskElemTy = hlfir::getFortranElementType(maskDef.getType());
    fir::LogicalType logicalFirType = {maskElemTy.dyn_cast<fir::LogicalType>()};
    kind = logicalFirType.getFKind();
    // Convert fir::LogicalType to mlir::Type
    logicalElemType = logicalFirType;
  }

  mlir::Operation *outputDef = args[0].getDefiningOp();
  mlir::Value outputAlloc = outputDef->getOperand(0);
  mlir::Type outType = hlfir::getFortranElementType(outputAlloc.getType());

  std::string fmfString{builder.getFastMathFlagsString()};
  std::string funcName =
      (mlir::Twine{callee.getLeafReference().getValue(), "x"} +
       mlir::Twine{rank} +
       (maskRank >= 0
            ? "_Logical" + mlir::Twine{kind} + "x" + mlir::Twine{maskRank}
            : "") +
       "_")
          .str();

  llvm::raw_string_ostream nameOS(funcName);
  outType.print(nameOS);
  nameOS << '_' << fmfString;

  auto typeGenerator = [rank](fir::FirOpBuilder &builder) {
    return genRuntimeMinlocType(builder, rank);
  };
  auto bodyGenerator = [rank, maskRank, inputType, logicalElemType, outType,
                        isMax](fir::FirOpBuilder &builder,
                               mlir::func::FuncOp &funcOp) {
    genRuntimeMinMaxlocBody(builder, funcOp, isMax, rank, maskRank, inputType,
                            logicalElemType, outType);
  };

  mlir::func::FuncOp newFunc =
      getOrCreateFunction(builder, funcName, typeGenerator, bodyGenerator);
  builder.create<fir::CallOp>(loc, newFunc,
                              mlir::ValueRange{args[0], args[1], args[5]});
  call->dropAllReferences();
  call->erase();
}

void SimplifyIntrinsicsPass::simplifyReductionBody(
    fir::CallOp call, const fir::KindMapping &kindMap,
    GenReductionBodyTy genBodyFunc, fir::FirOpBuilder &builder,
    const mlir::StringRef &funcName, mlir::Type elementType) {

  mlir::Operation::operand_range args = call.getArgs();

  mlir::Type resultType = call.getResult(0).getType();
  unsigned rank = getDimCount(args[0]);

  mlir::Location loc = call.getLoc();

  auto typeGenerator = [&resultType](fir::FirOpBuilder &builder) {
    return genNoneBoxType(builder, resultType);
  };
  auto bodyGenerator = [&rank, &genBodyFunc,
                        &elementType](fir::FirOpBuilder &builder,
                                      mlir::func::FuncOp &funcOp) {
    genBodyFunc(builder, funcOp, rank, elementType);
  };
  // Mangle the function name with the rank value as "x<rank>".
  mlir::func::FuncOp newFunc =
      getOrCreateFunction(builder, funcName, typeGenerator, bodyGenerator);
  auto newCall =
      builder.create<fir::CallOp>(loc, newFunc, mlir::ValueRange{args[0]});
  call->replaceAllUsesWith(newCall.getResults());
  call->dropAllReferences();
  call->erase();
}

void SimplifyIntrinsicsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::ModuleOp module = getOperation();
  fir::KindMapping kindMap = fir::getKindMapping(module);
  module.walk([&](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
      if (mlir::SymbolRefAttr callee = call.getCalleeAttr()) {
        mlir::StringRef funcName = callee.getLeafReference().getValue();
        // Replace call to runtime function for SUM when it has single
        // argument (no dim or mask argument) for 1D arrays with either
        // Integer4 or Real8 types. Other forms are ignored.
        // The new function is added to the module.
        //
        // Prototype for runtime call (from sum.cpp):
        // RTNAME(Sum<T>)(const Descriptor &x, const char *source, int line,
        //                int dim, const Descriptor *mask)
        //
        if (funcName.starts_with(RTNAME_STRING(Sum))) {
          simplifyIntOrFloatReduction(call, kindMap, genRuntimeSumBody);
          return;
        }
        if (funcName.starts_with(RTNAME_STRING(DotProduct))) {
          LLVM_DEBUG(llvm::dbgs() << "Handling " << funcName << "\n");
          LLVM_DEBUG(llvm::dbgs() << "Call operation:\n"; op->dump();
                     llvm::dbgs() << "\n");
          mlir::Operation::operand_range args = call.getArgs();
          const mlir::Value &v1 = args[0];
          const mlir::Value &v2 = args[1];
          mlir::Location loc = call.getLoc();
          fir::FirOpBuilder builder{getSimplificationBuilder(op, kindMap)};
          // Stringize the builder's FastMathFlags flags for mangling
          // the generated function name.
          std::string fmfString{builder.getFastMathFlagsString()};

          mlir::Type type = call.getResult(0).getType();
          if (!type.isa<mlir::FloatType>() && !type.isa<mlir::IntegerType>())
            return;

          // Try to find the element types of the boxed arguments.
          auto arg1Type = getArgElementType(v1);
          auto arg2Type = getArgElementType(v2);

          if (!arg1Type || !arg2Type)
            return;

          // Support only floating point and integer arguments
          // now (e.g. logical is skipped here).
          if (!arg1Type->isa<mlir::FloatType>() &&
              !arg1Type->isa<mlir::IntegerType>())
            return;
          if (!arg2Type->isa<mlir::FloatType>() &&
              !arg2Type->isa<mlir::IntegerType>())
            return;

          auto typeGenerator = [&type](fir::FirOpBuilder &builder) {
            return genRuntimeDotType(builder, type);
          };
          auto bodyGenerator = [&arg1Type,
                                &arg2Type](fir::FirOpBuilder &builder,
                                           mlir::func::FuncOp &funcOp) {
            genRuntimeDotBody(builder, funcOp, *arg1Type, *arg2Type);
          };

          // Suffix the function name with the element types
          // of the arguments.
          std::string typedFuncName(funcName);
          llvm::raw_string_ostream nameOS(typedFuncName);
          // We must mangle the generated function name with FastMathFlags
          // value.
          if (!fmfString.empty())
            nameOS << '_' << fmfString;
          nameOS << '_';
          arg1Type->print(nameOS);
          nameOS << '_';
          arg2Type->print(nameOS);

          mlir::func::FuncOp newFunc = getOrCreateFunction(
              builder, typedFuncName, typeGenerator, bodyGenerator);
          auto newCall = builder.create<fir::CallOp>(loc, newFunc,
                                                     mlir::ValueRange{v1, v2});
          call->replaceAllUsesWith(newCall.getResults());
          call->dropAllReferences();
          call->erase();

          LLVM_DEBUG(llvm::dbgs() << "Replaced with:\n"; newCall.dump();
                     llvm::dbgs() << "\n");
          return;
        }
        if (funcName.starts_with(RTNAME_STRING(Maxval))) {
          simplifyIntOrFloatReduction(call, kindMap, genRuntimeMaxvalBody);
          return;
        }
        if (funcName.starts_with(RTNAME_STRING(Count))) {
          simplifyLogicalDim0Reduction(call, kindMap, genRuntimeCountBody);
          return;
        }
        if (funcName.starts_with(RTNAME_STRING(Any))) {
          simplifyLogicalDim1Reduction(call, kindMap, genRuntimeAnyBody);
          return;
        }
        if (funcName.ends_with(RTNAME_STRING(All))) {
          simplifyLogicalDim1Reduction(call, kindMap, genRuntimeAllBody);
          return;
        }
        if (funcName.starts_with(RTNAME_STRING(Minloc))) {
          simplifyMinMaxlocReduction(call, kindMap, false);
          return;
        }
        if (funcName.starts_with(RTNAME_STRING(Maxloc))) {
          simplifyMinMaxlocReduction(call, kindMap, true);
          return;
        }
      }
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}

void SimplifyIntrinsicsPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  // LLVM::LinkageAttr creation requires that LLVM dialect is loaded.
  registry.insert<mlir::LLVM::LLVMDialect>();
}
std::unique_ptr<mlir::Pass> fir::createSimplifyIntrinsicsPass() {
  return std::make_unique<SimplifyIntrinsicsPass>();
}
