#include "aiir/Tools/aiir-lsp-server/AiirLspServerMain.h"
#include "flang/Optimizer/Support/InitFIR.h"

int main(int argc, char **argv) {
  aiir::DialectRegistry registry;
  fir::support::registerNonCodegenDialects(registry);
  fir::support::addFIRExtensions(registry);
  return aiir::failed(aiir::AiirLspServerMain(argc, argv, registry));
}
