#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "llvm/Support/TimeProfiler.h"

using namespace mlir;
using namespace mlir::cir;

namespace {

struct GotoSolverPass : public GotoSolverBase<GotoSolverPass> {

  GotoSolverPass() = default;
  void runOnOperation() override;
};

static void process(mlir::cir::FuncOp func) {

  mlir::OpBuilder rewriter(func.getContext());
  std::map<std::string, Block *> labels;
  std::vector<mlir::cir::GotoOp> gotos;

  func.getBody().walk([&](mlir::Operation *op) {
    if (auto lab = dyn_cast<mlir::cir::LabelOp>(op)) {
      labels.emplace(lab.getLabel().str(), lab->getBlock());
      lab.erase();
    } else if (auto goTo = dyn_cast<mlir::cir::GotoOp>(op)) {
      gotos.push_back(goTo);
    }
  });

  for (auto goTo : gotos) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(goTo);
    auto dest = labels[goTo.getLabel().str()];
    rewriter.create<mlir::cir::BrOp>(goTo.getLoc(), dest);
    goTo.erase();
  }
}

void GotoSolverPass::runOnOperation() {
  llvm::TimeTraceScope scope("Goto Solver");
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](mlir::cir::FuncOp op) { process(op); });
}

} // namespace

std::unique_ptr<Pass> mlir::createGotoSolverPass() {
  return std::make_unique<GotoSolverPass>();
}