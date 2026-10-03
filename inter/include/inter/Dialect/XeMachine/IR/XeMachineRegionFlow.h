#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGIONFLOW_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGIONFLOW_H

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace inter::xemachine {

class RegionFlow {
public:
  struct Transfer {
    mlir::Value input;
    mlir::OpOperand *operand = nullptr;
    mlir::Region *source = nullptr;
    mlir::Region *target = nullptr;
    mlir::Operation *sourceOperation = nullptr;
    unsigned successorInputIndex = 0;
  };

  struct Branch {
    llvm::SmallVector<mlir::Region *, 2> regions;
    llvm::SmallVector<Transfer, 8> transfers;
    llvm::SmallVector<llvm::BitVector, 2> reachable;
    llvm::BitVector entryRegions;
    llvm::BitVector repetitiveRegions;
    mlir::Operation *operation = nullptr;
  };

  explicit RegionFlow(mlir::Operation *root);

  llvm::ArrayRef<Branch> getBranches() const { return branches; }
  const Branch *lookup(mlir::Operation *operation) const;
  bool isRepetitive(mlir::Region *region) const;
  bool mayReach(mlir::Region *source, mlir::Region *target) const;
  bool areMutuallyExclusive(mlir::Region *lhs, mlir::Region *rhs) const;
  mlir::Region *getEnclosingRepetitiveRegion(mlir::Operation *operation) const;

private:
  struct RegionLocation {
    unsigned branch = 0;
    unsigned region = 0;
  };

  void build(mlir::RegionBranchOpInterface branch);

  llvm::SmallVector<Branch, 0> branches;
  llvm::DenseMap<mlir::Operation *, unsigned> branchIds;
  llvm::DenseMap<mlir::Region *, RegionLocation> regionLocations;
};

} // namespace inter::xemachine

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEREGIONFLOW_H
