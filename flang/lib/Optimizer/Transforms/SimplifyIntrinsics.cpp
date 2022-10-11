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

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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
      fir::FirOpBuilder &builder, mlir::func::FuncOp &funcOp, unsigned rank)>;

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
  /// Helper function to replace a reduction type of call with its
  /// simplified form. The actual function is generated using a callback
  /// function.
  /// \p call is the call to be replaced
  /// \p kindMap is used to create FIROpBuilder
  /// \p genBodyFunc is the callback that builds the replacement function
  void simplifyReduction(fir::CallOp call, const fir::KindMapping &kindMap,
                         GenReductionBodyTy genBodyFunc);
};

} // namespace

/// Generate function type for the simplified version of RTNAME(Sum) and
/// similar functions with a fir.box<none> type returning \p elementType.
static mlir::FunctionType genNoneBoxType(fir::FirOpBuilder &builder,
                                         const mlir::Type &elementType) {
  mlir::Type boxType = fir::BoxType::get(builder.getNoneType());
  return mlir::FunctionType::get(builder.getContext(), {boxType},
                                 {elementType});
}

using BodyOpGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &, mlir::Value,
    mlir::Value)>;
using InitValGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &)>;

/// Generate the reduction loop into \p funcOp.
///
/// \p initVal is a function, called to get the initial value for
///    the reduction value
/// \p genBody is called to fill in the actual reduciton operation
///    for example add for SUM, MAX for MAXVAL, etc.
/// \p rank is the rank of the input argument.
static void genReductionLoop(fir::FirOpBuilder &builder,
                             mlir::func::FuncOp &funcOp,
                             InitValGeneratorTy initVal,
                             BodyOpGeneratorTy genBody, unsigned rank) {
  auto loc = mlir::UnknownLoc::get(builder.getContext());
  mlir::Type elementType = funcOp.getResultTypes()[0];
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());

  mlir::IndexType idxTy = builder.getIndexType();

  mlir::Block::BlockArgListType args = funcOp.front().getArguments();
  mlir::Value arg = args[0];

  mlir::Value zeroIdx = builder.createIntegerConstant(loc, idxTy, 0);

  fir::SequenceType::Shape flatShape(rank,
                                     fir::SequenceType::getUnknownExtent());
  mlir::Type arrTy = fir::SequenceType::get(flatShape, elementType);
  mlir::Type boxArrTy = fir::BoxType::get(arrTy);
  mlir::Value array = builder.create<fir::ConvertOp>(loc, boxArrTy, arg);
  mlir::Value init = initVal(builder, loc, elementType);

  llvm::SmallVector<mlir::Value, 15> bounds;

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

  // Create a loop nest consisting of DoLoopOp operations.
  // Collect the loops' induction variables into indices array,
  // which will be used in the innermost loop to load the input
  // array's element.
  // The loops are generated such that the innermost loop processes
  // the 0 dimension.
  llvm::SmallVector<mlir::Value, 15> indices;
  for (unsigned i = rank; 0 < i; --i) {
    mlir::Value step = one;
    mlir::Value loopCount = bounds[i - 1];
    auto loop = builder.create<fir::DoLoopOp>(loc, zeroIdx, loopCount, step,
                                              /*unordered=*/false,
                                              /*finalCountValue=*/false, init);
    init = loop.getRegionIterArgs()[0];
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

  // Unwind the loop nest and insert ResultOp on each level
  // to return the updated value of the reduction to the enclosing
  // loops.
  for (unsigned i = 0; i < rank; ++i) {
    auto result = builder.create<fir::ResultOp>(loc, reductionVal);
    // Proceed to the outer loop.
    auto loop = mlir::cast<fir::DoLoopOp>(result->getParentOp());
    reductionVal = loop.getResult(0);
    // Set insertion point after the loop operation that we have
    // just processed.
    builder.setInsertionPointAfter(loop.getOperation());
  }

  // End of loop nest. The insertion point is after the outermost loop.
  // Return the reduction value from the function.
  builder.create<mlir::func::ReturnOp>(loc, reductionVal);
}

/// Generate function body of the simplified version of RTNAME(Sum)
/// with signature provided by \p funcOp. The caller is responsible
/// for saving/restoring the original insertion point of \p builder.
/// \p funcOp is expected to be empty on entry to this function.
/// \p rank specifies the rank of the input argument.
static void genRuntimeSumBody(fir::FirOpBuilder &builder,
                              mlir::func::FuncOp &funcOp, unsigned rank) {
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

  genReductionLoop(builder, funcOp, zero, genBodyOp, rank);
}

static void genRuntimeMaxvalBody(fir::FirOpBuilder &builder,
                                 mlir::func::FuncOp &funcOp, unsigned rank) {
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
    if (elementType.isa<mlir::FloatType>())
      return builder.create<mlir::arith::MaxFOp>(loc, elem1, elem2);
    if (elementType.isa<mlir::IntegerType>())
      return builder.create<mlir::arith::MaxSIOp>(loc, elem1, elem2);

    llvm_unreachable("unsupported type");
    return {};
  };
  genReductionLoop(builder, funcOp, init, genBodyOp, rank);
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

fir::ConvertOp expectConvertOp(mlir::Value val) {
  if (fir::ConvertOp op =
          mlir::dyn_cast_or_null<fir::ConvertOp>(val.getDefiningOp()))
    return op;
  LLVM_DEBUG(llvm::dbgs() << "Didn't find expected fir::ConvertOp\n");
  return nullptr;
}

static bool isOperandAbsent(mlir::Value val) {
  if (auto op = expectConvertOp(val)) {
    assert(op->getOperands().size() != 0);
    return mlir::isa_and_nonnull<fir::AbsentOp>(
        op->getOperand(0).getDefiningOp());
  }
  return false;
}

static bool isZero(mlir::Value val) {
  if (auto op = expectConvertOp(val)) {
    assert(op->getOperands().size() != 0);
    if (mlir::Operation *defOp = op->getOperand(0).getDefiningOp())
      return mlir::matchPattern(defOp, mlir::m_Zero());
  }
  return false;
}

static mlir::Value findBoxDef(mlir::Value val) {
  if (auto op = expectConvertOp(val)) {
    assert(op->getOperands().size() != 0);
    if (auto box = mlir::dyn_cast_or_null<fir::EmboxOp>(
            op->getOperand(0).getDefiningOp()))
      return box.getResult();
    if (auto box = mlir::dyn_cast_or_null<fir::ReboxOp>(
            op->getOperand(0).getDefiningOp()))
      return box.getResult();
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
/// \returns the element type or llvm::None if the type cannot
/// be reliably found.
/// We expect that the argument is a result of fir.convert
/// with the destination type of !fir.box<none>.
static llvm::Optional<mlir::Type> getArgElementType(mlir::Value val) {
  mlir::Operation *defOp;
  do {
    defOp = val.getDefiningOp();
    // Analyze only sequences of convert operations.
    if (!mlir::isa<fir::ConvertOp>(defOp))
      return llvm::None;
    val = defOp->getOperand(0);
    // The convert operation is expected to convert from one
    // box type to another box type.
    auto boxType = val.getType().cast<fir::BoxType>();
    auto elementType = fir::unwrapSeqOrBoxedSeqType(boxType);
    if (!elementType.isa<mlir::NoneType>())
      return elementType;
  } while (true);
}

void SimplifyIntrinsicsPass::simplifyReduction(fir::CallOp call,
                                               const fir::KindMapping &kindMap,
                                               GenReductionBodyTy genBodyFunc) {
  mlir::SymbolRefAttr callee = call.getCalleeAttr();
  mlir::Operation::operand_range args = call.getArgs();
  // args[1] and args[2] are source filename and line number, ignored.
  const mlir::Value &dim = args[3];
  const mlir::Value &mask = args[4];
  // dim is zero when it is absent, which is an implementation
  // detail in the runtime library.
  bool dimAndMaskAbsent = isZero(dim) && isOperandAbsent(mask);
  unsigned rank = getDimCount(args[0]);
  if (dimAndMaskAbsent && rank > 0) {
    mlir::Location loc = call.getLoc();
    fir::FirOpBuilder builder(call, kindMap);

    // Support only floating point and integer results now.
    mlir::Type resultType = call.getResult(0).getType();
    if (!resultType.isa<mlir::FloatType>() &&
        !resultType.isa<mlir::IntegerType>())
      return;

    auto argType = getArgElementType(args[0]);
    if (!argType)
      return;
    assert(*argType == resultType &&
           "Argument/result types mismatch in reduction");

    auto typeGenerator = [&resultType](fir::FirOpBuilder &builder) {
      return genNoneBoxType(builder, resultType);
    };
    auto bodyGenerator = [&rank, &genBodyFunc](fir::FirOpBuilder &builder,
                                               mlir::func::FuncOp &funcOp) {
      genBodyFunc(builder, funcOp, rank);
    };
    // Mangle the function name with the rank value as "x<rank>".
    std::string funcName =
        (mlir::Twine{callee.getLeafReference().getValue(), "x"} +
         mlir::Twine{rank})
            .str();
    mlir::func::FuncOp newFunc =
        getOrCreateFunction(builder, funcName, typeGenerator, bodyGenerator);
    auto newCall =
        builder.create<fir::CallOp>(loc, newFunc, mlir::ValueRange{args[0]});
    call->replaceAllUsesWith(newCall.getResults());
    call->dropAllReferences();
    call->erase();
  }
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
        if (funcName.startswith(RTNAME_STRING(Sum))) {
          simplifyReduction(call, kindMap, genRuntimeSumBody);
          return;
        }
        if (funcName.startswith(RTNAME_STRING(DotProduct))) {
          LLVM_DEBUG(llvm::dbgs() << "Handling " << funcName << "\n");
          LLVM_DEBUG(llvm::dbgs() << "Call operation:\n"; op->dump();
                     llvm::dbgs() << "\n");
          mlir::Operation::operand_range args = call.getArgs();
          const mlir::Value &v1 = args[0];
          const mlir::Value &v2 = args[1];
          mlir::Location loc = call.getLoc();
          fir::FirOpBuilder builder(op, kindMap);

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
          nameOS << "_";
          arg1Type->print(nameOS);
          nameOS << "_";
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
        if (funcName.startswith(RTNAME_STRING(Maxval))) {
          simplifyReduction(call, kindMap, genRuntimeMaxvalBody);
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
