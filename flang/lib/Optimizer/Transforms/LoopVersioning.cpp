//===- LoopVersioning.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass looks for loops iterating over assumed-shape arrays, that can
/// be optimized by "guessing" that the stride is element-sized.
///
/// This is done by createing two versions of the same loop: one which assumes
/// that the elements are contiguous (stride == size of element), and one that
/// is the original generic loop.
///
/// As a side-effect of the assumed element size stride, the array is also
/// flattened to make it a 1D array - this is because the internal array
/// structure must be either 1D or have known sizes in all dimensions - and at
/// least one of the dimensions here is already unknown.
///
/// There are two distinct benefits here:
/// 1. The loop that iterates over the elements is somewhat simplified by the
///    constant stride calculation.
/// 2. Since the compiler can understand the size of the stride, it can use
///    vector instructions, where an unknown (at compile time) stride does often
///    prevent vector operations from being used.
///
/// A known drawback is that the code-size is increased, in some cases that can
/// be quite substantial - 3-4x is quite plausible (this includes that the loop
/// gets vectorized, which in itself often more than doubles the size of the
/// code, because unless the loop size is known, there will be a modulo
/// vector-size remainder to deal with.
///
/// TODO: Do we need some size limit where loops no longer get duplicated?
//        Maybe some sort of cost analysis.
/// TODO: Should some loop content - for example calls to functions and
///       subroutines inhibit the versioning of the loops. Plausibly, this
///       could be part of the cost analysis above.
//===----------------------------------------------------------------------===//

#include "flang/ISO_Fortran_binding.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace fir {
#define GEN_PASS_DEF_LOOPVERSIONING
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-loop-versioning"

namespace {

class LoopVersioningPass
    : public fir::impl::LoopVersioningBase<LoopVersioningPass> {

public:
  void runOnOperation() override;
};

} // namespace

/// @c replaceOuterUses - replace uses outside of @c op with result of @c
/// outerOp
static void replaceOuterUses(mlir::Operation *op, mlir::Operation *outerOp) {
  const mlir::Operation *outerParent = outerOp->getParentOp();
  op->replaceUsesWithIf(outerOp, [&](mlir::OpOperand &operand) {
    mlir::Operation *owner = operand.getOwner();
    return outerParent == owner->getParentOp();
  });
}

static fir::SequenceType getAsSequenceType(mlir::Value *v) {
  mlir::Type argTy = fir::unwrapPassByRefType(fir::unwrapRefType(v->getType()));
  return argTy.dyn_cast<fir::SequenceType>();
}

void LoopVersioningPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::func::FuncOp func = getOperation();

  /// @c ArgInfo
  /// A structure to hold an argument, the size of the argument and dimension
  /// information.
  struct ArgInfo {
    mlir::Value *arg;
    size_t size;
    fir::BoxDimsOp dims[CFI_MAX_RANK];
  };

  // First look for arguments with assumed shape = unknown extent in the lowest
  // dimension.
  LLVM_DEBUG(llvm::dbgs() << "Func-name:" << func.getSymName() << "\n");
  mlir::Block::BlockArgListType args = func.getArguments();
  mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
  fir::KindMapping kindMap = fir::getKindMapping(module);
  mlir::SmallVector<ArgInfo> argsOfInterest;
  for (auto &arg : args) {
    if (auto seqTy = getAsSequenceType(&arg)) {
      unsigned rank = seqTy.getDimension();
      // Currently limited to 1D or 2D arrays as that seems to give good
      // improvement without excessive increase in code-size, etc.
      if (rank > 0 && rank < 3 &&
          seqTy.getShape()[0] == fir::SequenceType::getUnknownExtent()) {
        size_t typeSize = 0;
        mlir::Type elementType = fir::unwrapSeqOrBoxedSeqType(arg.getType());
        if (elementType.isa<mlir::FloatType>() ||
            elementType.isa<mlir::IntegerType>())
          typeSize = elementType.getIntOrFloatBitWidth() / 8;
        else if (auto cty = elementType.dyn_cast<fir::ComplexType>())
          typeSize = 2 * cty.getEleType(kindMap).getIntOrFloatBitWidth() / 8;
        if (typeSize)
          argsOfInterest.push_back({&arg, typeSize, {}});
        else
          LLVM_DEBUG(llvm::dbgs() << "Type not supported\n");

      } else {
        LLVM_DEBUG(llvm::dbgs() << "Too many dimensions\n");
      }
    }
  }

  if (argsOfInterest.empty())
    return;

  struct OpsWithArgs {
    mlir::Operation *op;
    mlir::SmallVector<ArgInfo> argsAndDims;
  };
  // Now see if those arguments are used inside any loop.
  mlir::SmallVector<OpsWithArgs, 4> loopsOfInterest;

  func.walk([&](fir::DoLoopOp loop) {
    mlir::Block &body = *loop.getBody();
    mlir::SmallVector<ArgInfo> argsInLoop;
    body.walk([&](fir::CoordinateOp op) {
      // The current operation could be inside another loop than
      // the one we're currently processing. Skip it, we'll get
      // to it later.
      if (op->getParentOfType<fir::DoLoopOp>() != loop)
        return;
      const mlir::Value &operand = op->getOperand(0);
      for (auto a : argsOfInterest) {
        if (*a.arg == operand) {
          // Only add if it's not already in the list.
          if (std::find_if(argsInLoop.begin(), argsInLoop.end(), [&](auto it) {
                return it.arg == a.arg;
              }) == argsInLoop.end()) {

            argsInLoop.push_back(a);
            break;
          }
        }
      }
    });

    if (!argsInLoop.empty()) {
      OpsWithArgs ops = {loop, argsInLoop};
      loopsOfInterest.push_back(ops);
    }
  });
  if (loopsOfInterest.empty())
    return;

  // If we get here, there are loops to process.
  fir::FirOpBuilder builder{module, kindMap};
  mlir::Location loc = builder.getUnknownLoc();
  mlir::IndexType idxTy = builder.getIndexType();

  LLVM_DEBUG(llvm::dbgs() << "Module Before transformation:");
  LLVM_DEBUG(module->dump());

  LLVM_DEBUG(llvm::dbgs() << "loopsOfInterest: " << loopsOfInterest.size()
                          << "\n");
  for (auto op : loopsOfInterest) {
    LLVM_DEBUG(op.op->dump());
    builder.setInsertionPoint(op.op);

    mlir::Value allCompares = nullptr;
    // Ensure all of the arrays are unit-stride.
    for (auto &arg : op.argsAndDims) {

      fir::SequenceType seqTy = getAsSequenceType(arg.arg);
      unsigned rank = seqTy.getDimension();

      // We only care about lowest order dimension.
      for (unsigned i = 0; i < rank; i++) {
        mlir::Value dimIdx = builder.createIntegerConstant(loc, idxTy, i);
        arg.dims[i] = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                     *arg.arg, dimIdx);
      }
      mlir::Value elemSize =
          builder.createIntegerConstant(loc, idxTy, arg.size);
      mlir::Value cmp = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, arg.dims[0].getResult(2),
          elemSize);
      if (!allCompares) {
        allCompares = cmp;
      } else {
        allCompares =
            builder.create<mlir::arith::AndIOp>(loc, cmp, allCompares);
      }
    }

    auto ifOp =
        builder.create<fir::IfOp>(loc, op.op->getResultTypes(), allCompares,
                                  /*withElse=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    LLVM_DEBUG(llvm::dbgs() << "Creating cloned loop\n");
    mlir::Operation *clonedLoop = op.op->clone();
    bool changed = false;
    for (auto &arg : op.argsAndDims) {
      fir::SequenceType::Shape newShape;
      newShape.push_back(fir::SequenceType::getUnknownExtent());
      auto elementType = fir::unwrapSeqOrBoxedSeqType(arg.arg->getType());
      mlir::Type arrTy = fir::SequenceType::get(newShape, elementType);
      mlir::Type boxArrTy = fir::BoxType::get(arrTy);
      mlir::Type refArrTy = builder.getRefType(arrTy);
      auto carg = builder.create<fir::ConvertOp>(loc, boxArrTy, *arg.arg);
      auto caddr = builder.create<fir::BoxAddrOp>(loc, refArrTy, carg);
      auto insPt = builder.saveInsertionPoint();
      // Use caddr instead of arg.
      clonedLoop->walk([&](fir::CoordinateOp coop) {
        // Reduce the multi-dimensioned index to a single index.
        // This is required becase fir arrays do not support multiple dimensions
        // with unknown dimensions at compile time.
        if (coop->getOperand(0) == *arg.arg &&
            coop->getOperands().size() >= 2) {
          builder.setInsertionPoint(coop);
          mlir::Value totalIndex = builder.createIntegerConstant(loc, idxTy, 0);
          // Operand(1) = array; Operand(2) = index1; Operand(3) = index2
          for (unsigned i = coop->getOperands().size() - 1; i > 1; i--) {
            mlir::Value curIndex =
                builder.createConvert(loc, idxTy, coop->getOperand(i));
            // First arg is Operand2, so dims[i-2] is 0-based i-1!
            mlir::Value scale =
                builder.createConvert(loc, idxTy, arg.dims[i - 2].getResult(1));
            totalIndex = builder.create<mlir::arith::AddIOp>(
                loc, totalIndex,
                builder.create<mlir::arith::MulIOp>(loc, scale, curIndex));
          }
          totalIndex = builder.create<mlir::arith::AddIOp>(
              loc, totalIndex,
              builder.createConvert(loc, idxTy, coop->getOperand(1)));

          auto newOp = builder.create<fir::CoordinateOp>(
              loc, builder.getRefType(elementType), caddr,
              mlir::ValueRange{totalIndex});
          LLVM_DEBUG(newOp->dump());
          coop->getResult(0).replaceAllUsesWith(newOp->getResult(0));
          coop->erase();
          changed = true;
        }
      });

      builder.restoreInsertionPoint(insPt);
    }
    assert(changed && "Expected operations to have changed");

    builder.insert(clonedLoop);
    // Forward the result(s), if any, from the loop operation to the
    //
    mlir::ResultRange results = clonedLoop->getResults();
    bool hasResults = (results.size() > 0);
    if (hasResults)
      builder.create<fir::ResultOp>(loc, results);

    // Add the original loop in the else-side of the if operation.
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    replaceOuterUses(op.op, ifOp);
    op.op->remove();
    builder.insert(op.op);
    // Rely on "cloned loop has results, so original loop also has results".
    if (hasResults) {
      builder.create<fir::ResultOp>(loc, op.op->getResults());
    } else {
      // Use an assert to check this.
      assert(op.op->getResults().size() == 0 &&
             "Weird, the cloned loop doesn't have results, but the original "
             "does?");
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "After transform:\n");
  LLVM_DEBUG(module->dump());

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}

std::unique_ptr<mlir::Pass> fir::createLoopVersioningPass() {
  return std::make_unique<LoopVersioningPass>();
}
