#ifndef INTER_EMIT_EMIT_H
#define INTER_EMIT_EMIT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class ModuleOp;
}

namespace inter {

mlir::LogicalResult emitIgaAsm(mlir::ModuleOp mod, llvm::raw_ostream &os);

} // namespace inter

#endif // INTER_EMIT_EMIT_H
