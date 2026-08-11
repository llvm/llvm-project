#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGALLOCPREPARATION_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGALLOCPREPARATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace inter::xemachine {

/// Makes destructive and region register aliases safe for register allocation.
/// This operation is idempotent.
mlir::LogicalResult prepareRegisterAllocation(mlir::func::FuncOp function);

} // namespace inter::xemachine

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGALLOCPREPARATION_H
