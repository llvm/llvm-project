#ifndef CLANG_CIR_PASSES_H
#define CLANG_CIR_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace cir {
namespace direct {
/// Create a pass that fully lowers CIR to the LLVMIR dialect.
std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass();

/// Adds passes that fully lower CIR to the LLVMIR dialect.
void populateCIRToLLVMPasses(mlir::OpPassManager &pm);

} // namespace direct
} // end namespace cir

#endif // CLANG_CIR_PASSES_H
