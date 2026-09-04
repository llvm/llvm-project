#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEALIASANALYSIS_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEALIASANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace inter::xemachine {

/// Immutable register-storage alias information for a function.
class RegisterAliasAnalysis {
public:
  /// A direct alias edge relative to the value used for the query.
  struct Alias {
    mlir::Value value;
    int64_t offsetDwords = 0;
    mlir::Operation *owner = nullptr;
    bool destructive = false;
  };

  /// A value's normalized location in an alias component.
  struct ValueInfo {
    unsigned component = 0;
    int64_t offsetDwords = 0;
  };

  /// A connected alias component in normalized dword coordinates.
  struct Component {
    llvm::SmallVector<mlir::Value, 4> members;
    int64_t widthDwords = 0;
    std::optional<int64_t> fixedOriginDwords;
  };

  RegisterAliasAnalysis() = default;

  static mlir::FailureOr<RegisterAliasAnalysis>
  create(mlir::func::FuncOp function);

  llvm::ArrayRef<mlir::Value> getValues() const;
  llvm::ArrayRef<Component> getComponents() const;
  const ValueInfo *lookup(mlir::Value value) const;
  llvm::ArrayRef<Alias> getAliases(mlir::Value value) const;

private:
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<Component> components;
  llvm::DenseMap<mlir::Value, ValueInfo> valueInfo;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<Alias, 4>> aliases;
};

} // namespace inter::xemachine

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEALIASANALYSIS_H
