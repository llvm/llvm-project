//===-- FIROpenMPOpsInterfaces.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements FIR operation interfaces, which may be attached
/// to OpenMP dialect operations.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROperationMoveOpInterface.h"
#include "flang/Optimizer/OpenMP/Support/RegisterOpenMPExtensions.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace {
/// Helper template that must be specialized for each operation.
/// The methods are declared just for documentation.
template <typename OP, typename Enable = void>
struct OperationMoveModel {
  // Returns true if it is allowed to move the given 'candidate'
  // operation from the 'descendant' operation into operation 'op'.
  // If 'candidate' is nullptr, then the caller is querying whether
  // any operation from any descendant can be moved into 'op' operation.
  bool canMoveFromDescendant(mlir::Operation *op, mlir::Operation *descendant,
                             mlir::Operation *candidate) const;

  // Returns true if it is allowed to move the given 'candidate'
  // operation out of operation 'op'. If 'candidate' is nullptr,
  // then the caller is querying whether any operation can be moved
  // out of 'op' operation.
  bool canMoveOutOf(mlir::Operation *op, mlir::Operation *candidate) const;
};

// Helpers to check if T is one of Ts.
template <typename T, typename... Ts>
struct is_any_type : std::disjunction<std::is_same<T, Ts>...> {};

template <typename T, typename... Ts>
struct is_any_omp_op
    : std::integral_constant<
          bool, is_any_type<typename std::remove_cv<T>::type, Ts...>::value> {};

template <typename T, typename... Ts>
constexpr bool is_any_omp_op_v = is_any_omp_op<T, Ts...>::value;

/// OperationMoveModel specialization for OMP_LOOP_WRAPPER_OPS.
template <typename OP>
struct OperationMoveModel<
    OP,
    typename std::enable_if<is_any_omp_op_v<OP, OMP_LOOP_WRAPPER_OPS>>::type>
    : public fir::OperationMoveOpInterface::ExternalModel<
          OperationMoveModel<OP>, OP> {
  bool canMoveFromDescendant(mlir::Operation *op, mlir::Operation *descendant,
                             mlir::Operation *candidate) const {
    // Operations cannot be moved from descendants of LoopWrapperInterface
    // operation into the LoopWrapperInterface operation.
    return false;
  }
  bool canMoveOutOf(mlir::Operation *op, mlir::Operation *candidate) const {
    // The LoopWrapperInterface operations are only supposed to contain
    // a loop operation, and it is probably okay to move operations
    // from the descendant loop operation out of the LoopWrapperInterface
    // operation. For now, return false to be conservative.
    return false;
  }
};

/// OperationMoveModel specialization for OMP_OUTLINEABLE_OPS.
template <typename OP>
struct OperationMoveModel<
    OP, typename std::enable_if<is_any_omp_op_v<OP, OMP_OUTLINEABLE_OPS>>::type>
    : public fir::OperationMoveOpInterface::ExternalModel<
          OperationMoveModel<OP>, OP> {
  bool canMoveFromDescendant(mlir::Operation *op, mlir::Operation *descendant,
                             mlir::Operation *candidate) const {
    // Operations can be moved from descendants of OutlineableOpenMPOpInterface
    // operation into the OutlineableOpenMPOpInterface operation.
    return true;
  }
  bool canMoveOutOf(mlir::Operation *op, mlir::Operation *candidate) const {
    // Operations cannot be moved out of OutlineableOpenMPOpInterface operation.
    return false;
  }
};

// Helper to call attachInterface<OperationMoveModel> for all Ts
// (types of operations).
template <typename... Ts>
void attachInterfaces(mlir::MLIRContext *ctx) {
  (Ts::template attachInterface<OperationMoveModel<Ts>>(*ctx), ...);
}
} // anonymous namespace

void fir::omp::registerOpInterfacesExtensions(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::omp::OpenMPDialect *dialect) {
        attachInterfaces<OMP_LOOP_WRAPPER_OPS>(ctx);
        attachInterfaces<OMP_OUTLINEABLE_OPS>(ctx);
      });
}
