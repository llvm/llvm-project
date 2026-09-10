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

  module->walk([&](Operation *operation) {
    auto aliasOp = dyn_cast<RegisterStorageAliasOpInterface>(operation);
    if (!aliasOp)
      return;
    SmallVector<RegisterStorageAlias> aliases;
    aliasOp.getRegisterStorageAliases(aliases);
    for (const RegisterStorageAlias &alias : aliases)
      llvm::outs() << operation->getName() << " offset=" << alias.offset
                   << " destructive=" << alias.destructive << "\n";
  });
  return 0;
}
