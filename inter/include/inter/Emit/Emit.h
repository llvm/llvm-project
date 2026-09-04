#ifndef INTER_EMIT_EMIT_H
#define INTER_EMIT_EMIT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir {
class ModuleOp;
}

namespace inter {

mlir::LogicalResult emitGedBinary(mlir::ModuleOp moduleOp,
                                  llvm::raw_ostream &output,
                                  uint32_t *payloadEntryOffset = nullptr);
mlir::LogicalResult emitAssembly(mlir::ModuleOp moduleOp,
                                 llvm::raw_ostream &output);
mlir::LogicalResult emitZebin(mlir::ModuleOp moduleOp,
                              llvm::raw_ostream &output);

} // namespace inter

#endif // INTER_EMIT_EMIT_H
