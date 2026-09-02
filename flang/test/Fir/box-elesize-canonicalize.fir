// RUN: fir-opt --canonicalize %s | FileCheck %s

// Fold: plain array box with dynamic extent — element type (f32) is statically known.
func.func @fold_array_f32(%arg0: !fir.box<!fir.array<?xf32>>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.box<!fir.array<?xf32>>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @fold_array_f32(
// CHECK:         %[[C:.*]] = arith.constant 4 : index
// CHECK:         return %[[C]]
// CHECK-NOT:     fir.box_elesize

// Fold: plain array box with static extent — element type (f64) is statically known.
func.func @fold_array_f64(%arg0: !fir.box<!fir.array<10xf64>>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.box<!fir.array<10xf64>>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @fold_array_f64(
// CHECK:         %[[C:.*]] = arith.constant 8 : index
// CHECK:         return %[[C]]
// CHECK-NOT:     fir.box_elesize

// Fold: scalar (non-array) box.
func.func @fold_scalar_i32(%arg0: !fir.box<i32>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.box<i32>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @fold_scalar_i32(
// CHECK:         %[[C:.*]] = arith.constant 4 : index
// CHECK:         return %[[C]]
// CHECK-NOT:     fir.box_elesize

// No fold: ClassType is polymorphic — the actual element type is only known at runtime.
func.func @no_fold_class(%arg0: !fir.class<!fir.type<sometype{i:i32}>>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.class<!fir.type<sometype{i:i32}>>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @no_fold_class(
// CHECK:         %[[V:.*]] = fir.box_elesize %{{.*}} : (!fir.class<!fir.type<sometype{i:i32}>>) -> index
// CHECK:         return %[[V]]

// No fold: !fir.box<none> is an assumed-type (TYPE(*)) — no static element type.
func.func @no_fold_none(%arg0: !fir.box<none>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.box<none>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @no_fold_none(
// CHECK:         %[[V:.*]] = fir.box_elesize %{{.*}} : (!fir.box<none>) -> index
// CHECK:         return %[[V]]

// Fold: allocatable array (!fir.box<!fir.heap<!fir.array<...>>>)
func.func @fold_heap_array(%arg0: !fir.box<!fir.heap<!fir.array<?xf32>>>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @fold_heap_array(
// CHECK:         %[[C:.*]] = arith.constant 4 : index
// CHECK:         return %[[C]]
// CHECK-NOT:     fir.box_elesize

// Fold: Fortran pointer array (!fir.box<!fir.ptr<!fir.array<...>>>)
func.func @fold_ptr_array(%arg0: !fir.box<!fir.ptr<!fir.array<?xf64>>>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.box<!fir.ptr<!fir.array<?xf64>>>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @fold_ptr_array(
// CHECK:         %[[C:.*]] = arith.constant 8 : index
// CHECK:         return %[[C]]
// CHECK-NOT:     fir.box_elesize

// No fold: dynamic-length CHARACTER array
func.func @no_fold_char_dynamic(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @no_fold_char_dynamic(
// CHECK:         %[[V:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
// CHECK:         return %[[V]]

// Fold: fixed-length CHARACTER array. Element size is statically known (5 bytes).
func.func @fold_char_static(%arg0: !fir.box<!fir.array<?x!fir.char<1,5>>>) -> index {
  %0 = fir.box_elesize %arg0 : (!fir.box<!fir.array<?x!fir.char<1,5>>>) -> index
  return %0 : index
}
// CHECK-LABEL: func.func @fold_char_static(
// CHECK:         %[[C:.*]] = arith.constant 5 : index
// CHECK:         return %[[C]]
// CHECK-NOT:     fir.box_elesize

// Tests inside a module with an explicit x86-64 DLTI data-layout spec.
module attributes { dlti.dl_spec = #dlti.dl_spec< i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {

  // fp80 edge case: x87 80-bit extended precision (Fortran REAL(10) on x86-64).
  // MLIR's DataLayout derives the natural byte size from the bit-width
  // (80 bits = 10 bytes) and the DLTI entry "f80 = dense<128> : vector<2xi64>"
  // gives ABI alignment = 16 bytes.
  func.func @fold_f80_with_dlti(%arg0: !fir.box<f80>) -> index {
    %0 = fir.box_elesize %arg0 : (!fir.box<f80>) -> index
    return %0 : index
  }
  // CHECK-LABEL: func.func @fold_f80_with_dlti(
  // CHECK:         %[[C:.*]] = arith.constant 16 : index
  // CHECK:         return %[[C]]
  // CHECK-NOT:     fir.box_elesize

  // Fold: packed derived type -- fir.type<tp<{i:i32,d:f64}>>.
  // This is the exact example from MattPD's review: on x86-64
  // (f64 ABI align=8B) the non-packed size is 16B and the packed
  // size is 12B (i32 allocSize=4B + f64 allocSize=alignTo(8,8)=8B,
  // no inter-field gap, no tail padding).
  // The element distance must be 12, not 16.
  func.func @fold_packed_record(%arg0: !fir.box<!fir.type<tp<{i:i32,d:f64}>>>) -> index {
    %0 = fir.box_elesize %arg0 : (!fir.box<!fir.type<tp<{i:i32,d:f64}>>>) -> index
    return %0 : index
  }
  // CHECK-LABEL: func.func @fold_packed_record(
  // CHECK:         %[[C:.*]] = arith.constant 12 : index
  // CHECK:         return %[[C]]
  // CHECK-NOT:     fir.box_elesize

  // Fold: packed derived type with a wider alignment gap.
  // On x86-64: i32 store=4B, complex<f64> store=16B.
  // Non-packed: alignTo(4,4)=4 + alignTo(4,8)+alignTo(16,8)=24, tail=24 -> 24B.
  // Packed:     4 + 16 = 20B, align=1.
  // The element distance must be 20, not 24.
  func.func @fold_packed_record2(%arg0: !fir.box<!fir.type<tp2<{i:i32,z:complex<f64>}>>>) -> index {
    %0 = fir.box_elesize %arg0 : (!fir.box<!fir.type<tp2<{i:i32,z:complex<f64>}>>>) -> index
    return %0 : index
  }
  // CHECK-LABEL: func.func @fold_packed_record2(
  // CHECK:         %[[C:.*]] = arith.constant 20 : index
  // CHECK:         return %[[C]]
  // CHECK-NOT:     fir.box_elesize

  // Fold: packed derived type with f80 component.
  // f80 on x86-64: store size = 10B, ABI alignment = 16B,
  //   allocation size = alignTo(10, 16) = 16B.
  // i8:  store size = 1B, ABI alignment = 1B, allocation size = 1B.
  // Packed size = 16 + 1 = 17B (not 10 + 1 = 11B).
  // This case distinguishes allocation-size from store-size in the packed path.
  func.func @fold_packed_f80_i8(%arg0: !fir.box<!fir.type<tp3<{x:f80,b:i8}>>>) -> index {
    %0 = fir.box_elesize %arg0 : (!fir.box<!fir.type<tp3<{x:f80,b:i8}>>>) -> index
    return %0 : index
  }
  // CHECK-LABEL: func.func @fold_packed_f80_i8(
  // CHECK:         %[[C:.*]] = arith.constant 17 : index
  // CHECK:         return %[[C]]
  // CHECK-NOT:     fir.box_elesize
}
