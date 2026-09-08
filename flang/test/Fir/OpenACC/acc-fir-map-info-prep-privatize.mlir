// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-fir-map-info-prep))" | FileCheck %s

// Privatized storage is wrapped by acc.map_info with byte size and map flags.
// The acc.privatize op remains for the storage handle, dynamic sizes, and
// parallelism levels.

// CHECK-LABEL: func.func @private_static_memref
// CHECK: %[[PRIV:.*]] = acc.privatize
// CHECK: %[[SIZE:.*]] = arith.constant 32 : i64
// CHECK: acc.map_info varPtr(%[[PRIV]] : !acc.private_type<memref<8xi32>>)
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(private)
func.func @private_static_memref() {
  %priv = acc.privatize par_dims(#acc<par_dims[]>)
      : () -> !acc.private_type<memref<8xi32>>
  return
}

// -----

// Parallel-level private flags are recorded on map_info alongside private.

// CHECK-LABEL: func.func @private_parallel_levels
// CHECK: %[[PRIV:.*]] = acc.privatize
// CHECK: acc.map_info varPtr(%[[PRIV]] : !acc.private_type<memref<8xi32>>)
// CHECK-SAME: mapFlags(private,gang_private,worker_private,vector_private)
func.func @private_parallel_levels() {
  %priv = acc.privatize
      par_dims(#acc<par_dims[block_x, thread_y, thread_x]>)
      : () -> !acc.private_type<memref<8xi32>>
  return
}

// -----

// Privatized storage need not name any parallel dimension - storage promoted to
// shared memory is private without being replicated per level. Such a
// privatization carries no par_dims at all, not an empty one.

// CHECK-LABEL: func.func @private_without_par_dims
// CHECK: %[[PRIV:.*]] = acc.privatize
// CHECK: acc.map_info varPtr(%[[PRIV]] : !acc.private_type<memref<8xi32>>)
// CHECK-SAME: mapFlags(private)
// CHECK-NOT: gang_private
func.func @private_without_par_dims() {
  %priv = acc.privatize : () -> !acc.private_type<memref<8xi32>>
  return
}

// -----

// A record element uses the padded stride: real(8) + real(4) has a size of 12
// and an alignment of 8, so consecutive elements are 16 bytes apart.

// CHECK-LABEL: func.func @private_static_record
// CHECK: %[[PRIV:.*]] = acc.privatize
// CHECK: %[[SIZE:.*]] = arith.constant 128 : i64
// CHECK: acc.map_info varPtr(%[[PRIV]]
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: elementSize(12)
// CHECK-SAME: mapFlags(private)
func.func @private_static_record() {
  %priv = acc.privatize par_dims(#acc<par_dims[]>)
      : () -> !acc.private_type<!fir.array<8x!fir.type<_QFTpair{hi:f64,lo:f32}>>>
  return
}

// -----

// A runtime extent is multiplied in, so the size is an SSA value rather than a
// constant. This is the case that has no memref equivalent: memref cannot hold
// a record element type.

// CHECK-LABEL: func.func @private_dynamic_record
// CHECK-SAME: %[[N:.*]]: index
// CHECK: %[[PRIV:.*]] = acc.privatize(%[[N]])
// CHECK: %[[STRIDE:.*]] = arith.constant 16 : i64
// CHECK: %[[EXTENT:.*]] = arith.index_cast %[[N]] : index to i64
// CHECK: %[[SIZE:.*]] = arith.muli %[[STRIDE]], %[[EXTENT]] : i64
// CHECK: acc.map_info varPtr(%[[PRIV]]
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: mapFlags(private)
func.func @private_dynamic_record(%n: index) {
  %priv = acc.privatize(%n) par_dims(#acc<par_dims[]>)
      : (index) -> !acc.private_type<!fir.heap<!fir.array<?x!fir.type<_QFTpair{hi:f64,lo:f32}>>>>
  return
}

// -----

// Both extents of a partially dynamic shape contribute: the static extent is
// folded into the stride constant, the dynamic one is multiplied in.

// CHECK-LABEL: func.func @private_mixed_extents
// CHECK-SAME: %[[N:.*]]: index
// CHECK: %[[PRIV:.*]] = acc.privatize(%[[N]])
// CHECK: %[[STRIDE:.*]] = arith.constant 16 : i64
// CHECK: %[[EXTENT:.*]] = arith.index_cast %[[N]] : index to i64
// CHECK: %[[SIZE:.*]] = arith.muli %[[STRIDE]], %[[EXTENT]] : i64
// CHECK: acc.map_info varPtr(%[[PRIV]]
// CHECK-SAME: size(%[[SIZE]] : i64)
func.func @private_mixed_extents(%n: index) {
  %priv = acc.privatize(%n) par_dims(#acc<par_dims[]>)
      : (index) -> !acc.private_type<!fir.heap<!fir.array<4x?xf32>>>
  return
}

// -----

// A descriptor carries its own extents, so the type does not describe the
// storage and no map_info is created for it.

// CHECK-LABEL: func.func @private_descriptor
// CHECK: acc.privatize
// CHECK-NOT: acc.map_info
func.func @private_descriptor() {
  %priv = acc.privatize par_dims(#acc<par_dims[]>)
      : () -> !acc.private_type<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  return
}
