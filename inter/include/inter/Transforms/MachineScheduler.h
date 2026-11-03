//===- MachineScheduler.h - Generic machine scheduling ---------*- C++ -*-===//

#ifndef INTER_TRANSFORMS_MACHINESCHEDULER_H
#define INTER_TRANSFORMS_MACHINESCHEDULER_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <memory>

namespace mlir {
class Operation;
class Region;
class Value;
} // namespace mlir

namespace inter {

enum class MachineHazardKind : uint8_t { raw, war, waw, order };

struct MachineScheduleIssue {
  int64_t cycle = 0;
  int64_t nextCycle = 0;
  uint64_t stallCycles = 0;
  bool instruction = false;
};

struct MachineScheduleDependency {
  mlir::Operation *producer = nullptr;
  MachineHazardKind kind = MachineHazardKind::raw;
  MachineScheduleIssue issue;
};

struct MachineStorageAccess {
  uint64_t resource = 0;
  bool read = false;
  bool write = false;
};

struct MachineExtraDependency {
  mlir::Operation *source = nullptr;
  mlir::Operation *target = nullptr;
  MachineHazardKind kind = MachineHazardKind::raw;
};

class MachineScheduleState {
public:
  virtual ~MachineScheduleState() = default;

  virtual mlir::FailureOr<MachineScheduleIssue> previewIssue(
      mlir::Operation *operation,
      llvm::ArrayRef<MachineScheduleDependency> dependencies) const = 0;
  virtual mlir::FailureOr<MachineScheduleIssue>
  commitIssue(mlir::Operation *operation,
              llvm::ArrayRef<MachineScheduleDependency> dependencies) = 0;
  virtual mlir::FailureOr<bool>
  canFill(mlir::Operation *baseline,
          llvm::ArrayRef<MachineScheduleDependency> baselineDependencies,
          mlir::Operation *candidate,
          llvm::ArrayRef<MachineScheduleDependency> candidateDependencies)
      const = 0;
};

class MachineScheduleRegionSession {
public:
  virtual ~MachineScheduleRegionSession() = default;

  virtual bool canSchedulePrefix(llvm::ArrayRef<unsigned> prefix) const = 0;
};

class MachineScheduleModel {
public:
  virtual ~MachineScheduleModel() = default;

  virtual bool isSchedulable(mlir::Operation *operation) const = 0;
  virtual bool isNoInstruction(mlir::Operation *operation) const = 0;
  virtual bool isSupportedRegionOperation(mlir::Operation *operation) const = 0;
  virtual MachineHazardKind
  classifyDataDependency(mlir::Value operand) const = 0;
  virtual void getStorageAccesses(
      mlir::Operation *operation,
      llvm::SmallVectorImpl<MachineStorageAccess> &accesses) const = 0;
  virtual void getExtraDependencies(
      llvm::ArrayRef<mlir::Operation *> operations,
      llvm::SmallVectorImpl<MachineExtraDependency> &dependencies) const = 0;
  virtual std::unique_ptr<MachineScheduleRegionSession>
  createRegionSession(llvm::ArrayRef<mlir::Operation *> operations) const = 0;
  virtual std::unique_ptr<MachineScheduleState> createState() const = 0;
};

mlir::LogicalResult scheduleMachineRegion(mlir::Region &region,
                                          const MachineScheduleModel &model);

} // namespace inter

#endif // INTER_TRANSFORMS_MACHINESCHEDULER_H
