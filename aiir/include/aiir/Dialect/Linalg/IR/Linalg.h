//===- Linalg.h - Linalg dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LINALG_IR_LINALG_H
#define AIIR_DIALECT_LINALG_IR_LINALG_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/Dialect/Utils/ReshapeOpsUtils.h"
#include "aiir/Dialect/Utils/StructuredOpsUtils.h"
#include "aiir/IR/AffineExpr.h"
#include "aiir/IR/AffineMap.h"
#include "aiir/IR/BuiltinDialect.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/ImplicitLocOpBuilder.h"
#include "aiir/IR/TypeUtilities.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/DestinationStyleOpInterface.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/TilingInterface.h"
#include "aiir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>

namespace aiir {
namespace linalg {

class LinalgOp;

/// Returns the name mangled library call name to disambiguate between different
/// overloads at the C level. The name mangling scheme is basic and uses AIIR
/// type names:
///   1. form a string which is the concatenation of the linalg op name with all
///      the operand type names, separate by underscores;
///   2. drop the `linalg.` prefix, and the `<`, `>`, `?` symbols from the type.
/// Assumes `op` is a LinalgOp.
///
/// Examples:
///
/// 1. linalg.fill(%f, %A) : f32, memref<f32>
///   name mangles into `linalg_fill_f32_viewf32`
///
/// 2. linalg.dot %A, %B, %C :
///      (memref<?xf32, stride_specification>,
///       memref<?xf32, stride_specification>, memref<f32>)
///   name mangles into `linalg_dot_viewxf32_viewxf32_viewf32`
///
/// 3. linalg.matmul(...) :
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>
///   name mangles into `linalg_matmul_viewxxf32_viewxxf32_viewxxf32`
std::string generateLibraryCallName(Operation *op);

/// Returns `num` AffineDimExpr dimensions at positions
///   [startIdx, startIdx + num) and increments `startIdx` to `startIdx + num`.
SmallVector<AffineExpr, 4> makeAffineDimExprs(unsigned num, unsigned &startIdx,
                                              AIIRContext *context);

/// Returns `maybeMap.get()` if `maybeMap` is set, otherwise returns the
/// symbol-less identity map of `rank`.
AffineMap extractOrIdentityMap(std::optional<AffineMap> maybeMap, unsigned rank,
                               AIIRContext *context);

/// Return the vector that is the concatenation of `a` and `b`.
SmallVector<AffineExpr, 4> concat(ArrayRef<AffineExpr> a,
                                  ArrayRef<AffineExpr> b);

/// Create one memref::DimOp or tensor::DimOp depending on the type of `val`.
/// This is a polymorphic convenience function to abstract away the rank and
/// concrete type of `val`.
/// Asserts that `val` is a memref or tensor type.
Value createOrFoldDimOp(OpBuilder &b, Location loc, Value val, int64_t dim);

/// Create one memref::DimOp or tensor::DimOp depending on the type of `val`.
/// This is a polymorphic convenience function to abstract away the rank and
/// concrete type of `val`.
/// Asserts that `val` is a memref or tensor type.
OpFoldResult createFoldedDimOp(OpBuilder &b, Location loc, Value val,
                               int64_t dim);

} // namespace linalg
} // namespace aiir

//===----------------------------------------------------------------------===//
// Linalg Dialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Enums
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Linalg/IR/LinalgOpsEnums.h.inc"

namespace aiir {
namespace linalg {

/// Converts the given `m` and `r` parameters to a WinogradConv2DFmr enumeration
/// value.
std::optional<WinogradConv2DFmr> getWinogradConv2DFmr(int64_t m, int64_t r);

/// Converts the given WinogradConv2DFmr enumeration value to a pair of
/// m and r parameters.
std::pair<int64_t, int64_t> getFmrFromWinogradConv2DFmr(WinogradConv2DFmr fmr);

} // namespace linalg
} // namespace aiir

//===----------------------------------------------------------------------===//
// Linalg Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/Linalg/IR/LinalgOpsAttrDefs.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Interfaces
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Linalg/IR/LinalgInterfaces.h"

//===----------------------------------------------------------------------===//
// Linalg Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Linalg/IR/LinalgOps.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/Linalg/IR/LinalgRelayoutOps.h.inc"

namespace aiir::linalg {

/// Returns the outer shape in the packed domain before applying the
/// transposition.
template <typename OpTy,
          typename = std::enable_if_t<std::is_same_v<OpTy, linalg::PackOp> ||
                                      std::is_same_v<OpTy, linalg::UnPackOp>>>
SmallVector<int64_t> getPackedOuterShapeWithoutTransposition(OpTy packOrUnPack);

/// Specialization of `linalg.matmul` op that has a transpose map on A
class MatmulTransposeAOp : public MatmulOp {
  /// Create an affine map for a transpose-A matmul. Used only in the builders.
  static SmallVector<AffineMap> getDefaultIndexingMaps(OpBuilder &builder);

public:
  using MatmulOp::MatmulOp;
  static ::aiir::TypeID resolveTypeID() { return TypeID::get<MatmulOp>(); }

  /// Build a transpose A matmul.
  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeAOp create(OpBuilder &builder, Location location,
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose A matmul with a specific result type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeAOp create(OpBuilder &builder, Location location,
                                   TypeRange resultTensorTypes,
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose A matmul with a specific result type and a cast type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs, Attribute cast,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeAOp create(OpBuilder &builder, Location location,
                                   TypeRange resultTensorTypes,
                                   ValueRange inputs, ValueRange outputs,
                                   Attribute cast,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Checks if the affine map is the expected one for this operation
  static bool isDefaultIndexingMaps(Attribute attr);

  static bool classof(Operation *op);
};

/// Specialization of `linalg.matmul` op that has a transpose map on B
class MatmulTransposeBOp : public MatmulOp {
  /// Create an affine map for a transpose-B matmul. Used only in the builders.
  static SmallVector<AffineMap> getDefaultIndexingMaps(OpBuilder &builder);

public:
  using MatmulOp::MatmulOp;
  static ::aiir::TypeID resolveTypeID() { return TypeID::get<MatmulOp>(); }

  /// Build a transpose B matmul.
  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeBOp create(OpBuilder &builder, Location location,
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose B matmul with a specific result type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeBOp create(OpBuilder &builder, Location location,
                                   TypeRange resultTensorTypes,
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose B matmul with a specific result type and a cast type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs, Attribute cast,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeBOp create(OpBuilder &builder, Location location,
                                   TypeRange resultTensorTypes,
                                   ValueRange inputs, ValueRange outputs,
                                   Attribute cast,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Checks if the affine map is the expected one for this operation
  static bool isDefaultIndexingMaps(Attribute attr);

  static bool classof(Operation *op);
};

/// Specialization of `linalg.batch_matmul` op that has a transpose map on A
class BatchMatmulTransposeAOp : public BatchMatmulOp {
  /// Create an affine map for a transpose-A batch_matmul. Used only in the
  /// builders.
  static SmallVector<AffineMap> getDefaultIndexingMaps(OpBuilder &builder);

public:
  using BatchMatmulOp::BatchMatmulOp;
  static ::aiir::TypeID resolveTypeID() { return TypeID::get<BatchMatmulOp>(); }

  /// Build a transpose A matmul.
  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeAOp
  create(OpBuilder &builder, Location location, ValueRange inputs,
         ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose A matmul with a specific result type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeAOp
  create(OpBuilder &builder, Location location, TypeRange resultTensorTypes,
         ValueRange inputs, ValueRange outputs,
         ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose A matmul with a specific result type and a cast type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs, Attribute cast,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeAOp
  create(OpBuilder &builder, Location location, TypeRange resultTensorTypes,
         ValueRange inputs, ValueRange outputs, Attribute cast,
         ArrayRef<NamedAttribute> attributes = {});

  /// Checks if the affine map is the expected one for this operation
  static bool isDefaultIndexingMaps(Attribute attr);

  static bool classof(Operation *op);
};

/// Specialization of `linalg.batch_matmul` op that has a transpose map on B
class BatchMatmulTransposeBOp : public BatchMatmulOp {
  /// Create an affine map for a transpose-B batch_matmul. Used only in the
  /// builders.
  static SmallVector<AffineMap> getDefaultIndexingMaps(OpBuilder &builder);

public:
  using BatchMatmulOp::BatchMatmulOp;
  static ::aiir::TypeID resolveTypeID() { return TypeID::get<BatchMatmulOp>(); }

  /// Build a transpose B matmul.
  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeBOp
  create(OpBuilder &builder, Location location, ValueRange inputs,
         ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose B matmul with a specific result type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeBOp
  create(OpBuilder &builder, Location location, TypeRange resultTensorTypes,
         ValueRange inputs, ValueRange outputs,
         ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose B matmul with a specific result type and a cast type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs, Attribute cast,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeBOp
  create(OpBuilder &builder, Location location, TypeRange resultTensorTypes,
         ValueRange inputs, ValueRange outputs, Attribute cast,
         ArrayRef<NamedAttribute> attributes = {});

  /// Checks if the affine map is the expected one for this operation
  static bool isDefaultIndexingMaps(Attribute attr);

  static bool classof(Operation *op);
};

} // namespace aiir::linalg

#endif // AIIR_DIALECT_LINALG_IR_LINALG_H
