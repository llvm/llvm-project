#include "inter/Compiler/Compiler.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  inter::registerCompilerPasses();
  mlir::DialectRegistry registry;
  inter::registerCompilerDialects(registry);
  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "inter-opt: inter dialect tool\n", registry));
}
