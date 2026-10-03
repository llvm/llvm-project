#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGALLOCPREPARATION_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGALLOCPREPARATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace inter::xemachine {

class UpdateTupleOp;

/// Makes destructive and region register aliases safe for register allocation.
/// This operation is idempotent.
mlir::LogicalResult prepareRegisterAllocation(mlir::func::FuncOp function);

/// Returns whether the update consumes a prepared update-base copy.
bool hasPreparedUpdateBaseCopy(UpdateTupleOp update);

} // namespace inter::xemachine

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGALLOCPREPARATION_H
