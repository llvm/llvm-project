#ifndef INTER_COMPILER_COMPILER_H
#define INTER_COMPILER_COMPILER_H

#include "inter/Dialect/XeMachine/IR/XeMachineTarget.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace llvm {
class Module;
class raw_ostream;
} // namespace llvm

namespace mlir {
class DialectRegistry;
}

namespace inter {

enum class CompilationOutput { zebin, ged, assembly, none };

struct CompilerOptions {
  xemachine::TargetConfig target =
      xemachine::TargetConfig::get(xemachine::TargetChip::bmg);
  unsigned simdWidth = 16;
  std::string transformLibraryPath;
  CompilationOutput output = CompilationOutput::zebin;
};

void registerCompilerDialects(mlir::DialectRegistry &registry);
void registerCompilerPasses();

llvm::Error compileLLVMModule(std::unique_ptr<llvm::Module> llvmModule,
                              llvm::raw_ostream &output,
                              llvm::raw_ostream &diagnosticOutput,
                              const CompilerOptions &options);

} // namespace inter

#endif // INTER_COMPILER_COMPILER_H
