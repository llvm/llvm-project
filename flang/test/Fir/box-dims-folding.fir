// RUN: fir-opt --canonicalize %s -split-input-file | FileCheck %s

// fir.box_dims of a fir.create_box with a constant dimension forwards the lower
// bound, extent, and byte stride operands that built that dimension.
// CHECK-LABEL: func.func @fold_box_dims_create_box(
// CHECK-SAME:      %[[BASE:[^:]+]]: !fir.ref<!fir.array<?xf32>>, %[[LB:[^:]+]]: index, %[[EXT:[^:]+]]: index, %[[SM:[^:]+]]: index)
// CHECK-NOT:       fir.box_dims
// CHECK:           return %[[LB]], %[[EXT]], %[[SM]]
func.func @fold_box_dims_create_box(%base: !fir.ref<!fir.array<?xf32>>, %lb: index, %ext: index, %sm: index) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %0 = fir.create_box %base lbs(%lb) extents(%ext) strides(%sm) : (!fir.ref<!fir.array<?xf32>>, index, index, index) -> !fir.box<!fir.array<?xf32>>
  %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// The forwarded operands are selected per dimension: dim 1 forwards the second
// lower bound / extent / stride triple.
// CHECK-LABEL: func.func @fold_box_dims_create_box_2d(
// CHECK-SAME:      %[[BASE:[^:]+]]: !fir.ref<!fir.array<?x?xf32>>, %[[LB0:[^:]+]]: index, %[[EXT0:[^:]+]]: index, %[[SM0:[^:]+]]: index, %[[LB1:[^:]+]]: index, %[[EXT1:[^:]+]]: index, %[[SM1:[^:]+]]: index)
// CHECK-NOT:       fir.box_dims
// CHECK:           return %[[LB1]], %[[EXT1]], %[[SM1]]
func.func @fold_box_dims_create_box_2d(%base: !fir.ref<!fir.array<?x?xf32>>, %lb0: index, %ext0: index, %sm0: index, %lb1: index, %ext1: index, %sm1: index) -> (index, index, index) {
  %c1 = arith.constant 1 : index
  %0 = fir.create_box %base lbs(%lb0, %lb1) extents(%ext0, %ext1) strides(%sm0, %sm1) : (!fir.ref<!fir.array<?x?xf32>>, index, index, index, index, index, index) -> !fir.box<!fir.array<?x?xf32>>
  %1:3 = fir.box_dims %0, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// The create_box operands are AnyIntegerType; when they are not index-typed the
// fold inserts fir.convert so the forwarded values match the box_dims results.
// CHECK-LABEL: func.func @fold_box_dims_create_box_i64_operands(
// CHECK-SAME:      %[[BASE:[^:]+]]: !fir.ref<!fir.array<?xf32>>, %[[LB:[^:]+]]: i64, %[[EXT:[^:]+]]: i64, %[[SM:[^:]+]]: i64)
// CHECK-NOT:       fir.box_dims
// CHECK-DAG:       %[[LBC:.*]] = fir.convert %[[LB]] : (i64) -> index
// CHECK-DAG:       %[[EXTC:.*]] = fir.convert %[[EXT]] : (i64) -> index
// CHECK-DAG:       %[[SMC:.*]] = fir.convert %[[SM]] : (i64) -> index
// CHECK:           return %[[LBC]], %[[EXTC]], %[[SMC]]
func.func @fold_box_dims_create_box_i64_operands(%base: !fir.ref<!fir.array<?xf32>>, %lb: i64, %ext: i64, %sm: i64) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %0 = fir.create_box %base lbs(%lb) extents(%ext) strides(%sm) : (!fir.ref<!fir.array<?xf32>>, i64, i64, i64) -> !fir.box<!fir.array<?xf32>>
  %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// No fold: the queried dimension is not a compile-time constant, so the right
// operand triple cannot be selected.
// CHECK-LABEL: func.func @no_fold_box_dims_create_box_dynamic_dim(
// CHECK:           %[[BOX:.*]] = fir.create_box
// CHECK:           fir.box_dims %[[BOX]]
func.func @no_fold_box_dims_create_box_dynamic_dim(%base: !fir.ref<!fir.array<?x?xf32>>, %lb0: index, %ext0: index, %sm0: index, %lb1: index, %ext1: index, %sm1: index, %d: index) -> (index, index, index) {
  %0 = fir.create_box %base lbs(%lb0, %lb1) extents(%ext0, %ext1) strides(%sm0, %sm1) : (!fir.ref<!fir.array<?x?xf32>>, index, index, index, index, index, index) -> !fir.box<!fir.array<?x?xf32>>
  %1:3 = fir.box_dims %0, %d : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  return %1#0, %1#1, %1#2 : index, index, index
}
