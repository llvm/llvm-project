#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace inter::xemachine;

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: " << argv[0] << " <input file>\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<XeMachineDialect, func::FuncDialect>();
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module)
    return 1;

  WalkResult walk = module->walk([&](Operation *operation) {
    if (!isa<InstructionIssueOpInterface>(operation))
      return WalkResult::advance();
    FailureOr<Xe2InstructionTiming> timing = getXe2InstructionTiming(operation);
    if (failed(timing))
      return WalkResult::interrupt();
    llvm::outs() << operation->getName() << " class="
                 << stringifyInstructionIssueClass(timing->issueClass)
                 << " pipe=" << stringifyXe2IssuePipe(timing->pipe)
                 << " latency=" << timing->completionLatency
                 << " occupancy=" << timing->occupancy;
    if (timing->sendSourceReadLatency)
      llvm::outs() << " send-read=" << *timing->sendSourceReadLatency;
    llvm::outs() << " raw-gap="
                 << getXe2RequiredGap(*timing, Xe2DependencyKind::raw)
                 << " war-gap="
                 << getXe2RequiredGap(*timing, Xe2DependencyKind::war)
                 << " order-gap="
                 << getXe2RequiredGap(*timing, Xe2DependencyKind::order)
                 << "\n";
    return WalkResult::advance();
  });
  return walk.wasInterrupted();
}
