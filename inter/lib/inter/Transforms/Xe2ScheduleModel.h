#ifndef INTER_LIB_TRANSFORMS_XE2SCHEDULEMODEL_H
#define INTER_LIB_TRANSFORMS_XE2SCHEDULEMODEL_H

#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir::func {
class FuncOp;
} // namespace mlir::func

namespace inter {
class MachineScheduleModel;

mlir::FailureOr<std::unique_ptr<MachineScheduleModel>>
createXe2ScheduleModel(mlir::func::FuncOp function);

} // namespace inter

#endif // INTER_LIB_TRANSFORMS_XE2SCHEDULEMODEL_H
