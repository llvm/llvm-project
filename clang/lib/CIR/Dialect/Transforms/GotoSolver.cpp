#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>

using namespace mlir;
using namespace cir;

namespace {

struct GotoSolverPass : public GotoSolverBase<GotoSolverPass> {

  GotoSolverPass() = default;
  void runOnOperation() override;
};

static void process(cir::FuncOp func) {

  mlir::OpBuilder rewriter(func.getContext());
  llvm::StringMap<Block *> labels;
  llvm::SmallVector<cir::GotoOp, 4> gotos;

  func.getBody().walk([&](mlir::Operation *op) {
    if (auto lab = dyn_cast<cir::LabelOp>(op)) {
      // Will construct a string copy inplace. Safely erase the label
      labels.try_emplace(lab.getLabel(), lab->getBlock());
      lab.erase();
    } else if (auto goTo = dyn_cast<cir::GotoOp>(op)) {
      gotos.push_back(goTo);
    }
  });

  for (auto goTo : gotos) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(goTo);
    Block *dest = labels[goTo.getLabel()];
    rewriter.create<cir::BrOp>(goTo.getLoc(), dest);
    goTo.erase();
  }
}

void GotoSolverPass::runOnOperation() {
  llvm::TimeTraceScope scope("Goto Solver");
  getOperation()->walk([&](cir::FuncOp op) { process(op); });
}

} // namespace

std::unique_ptr<Pass> mlir::createGotoSolverPass() {
  return std::make_unique<GotoSolverPass>();
}
