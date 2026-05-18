#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "flang/Optimizer/Support/InitFIR.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  fir::support::registerNonCodegenDialects(registry);
  fir::support::addFIRExtensions(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
