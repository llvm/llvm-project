// RUN: fir-opt --split-input-file --fir-to-llvm-ir %s | FileCheck %s

// Test simple global LLVM conversion

fir.global @g_i0 : i32 {
  %1 = arith.constant 0 : i32
  fir.has_value %1 : i32
}

// CHECK: llvm.mlir.global external @g_i0() : i32 {
// CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   llvm.return %[[C0]] : i32
// CHECK: }

// -----

fir.global @g_ci5 constant : i32 {
  %c = arith.constant 5 : i32
  fir.has_value %c : i32
}

// CHECK: llvm.mlir.global external constant @g_ci5() : i32 {
// CHECK:   %[[C5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:   llvm.return %[[C5]] : i32
// CHECK: }

// -----

fir.global internal @i_i515 (515:i32) : i32
// CHECK: llvm.mlir.global internal @i_i515(515 : i32) : i32

// -----

fir.global common @C_i511 (0:i32) : i32
// CHECK: llvm.mlir.global common @C_i511(0 : i32) : i32

// -----

fir.global weak @w_i86 (86:i32) : i32
// CHECK: llvm.mlir.global weak @w_i86(86 : i32) : i32

// -----

fir.global linkonce @w_i86 (86:i32) : i32
// CHECK: llvm.mlir.global linkonce @w_i86(86 : i32) : i32

// -----

// Test conversion of fir.address_of with fir.global

func @f1() {
  %0 = fir.address_of(@symbol) : !fir.ref<i64>
  return
}

fir.global @symbol : i64 {
  %0 = arith.constant 1 : i64
  fir.has_value %0 : i64
}

// CHECK: %{{.*}} = llvm.mlir.addressof @[[SYMBOL:.*]] : !llvm.ptr<i64>

// CHECK: llvm.mlir.global external @[[SYMBOL]]() : i64 {
// CHECK:   %{{.*}} = llvm.mlir.constant(1 : i64) : i64
// CHECK:   llvm.return %{{.*}} : i64
// CHECK: }

// -----

// Test global with insert_on_range operation covering the full array
// in initializer region.

fir.global internal @_QEmultiarray : !fir.array<32x32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32x32xi32>
  %2 = fir.insert_on_range %0, %c0_i32, [0 : index, 31 : index, 0 : index, 31 : index] : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
  fir.has_value %2 : !fir.array<32x32xi32>
}

// CHECK: llvm.mlir.global internal @_QEmultiarray() : !llvm.array<32 x array<32 x i32>> {
// CHECK:   %[[CST:.*]] = llvm.mlir.constant(dense<1> : vector<32x32xi32>) : !llvm.array<32 x array<32 x i32>>
// CHECK:   llvm.return %[[CST]] : !llvm.array<32 x array<32 x i32>>
// CHECK: }

// -----

// Test global with insert_on_range operation not covering the full array
// in initializer region.

fir.global internal @_QEmultiarray : !fir.array<32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32xi32>
  %2 = fir.insert_on_range %0, %c0_i32, [5 : index, 31 : index] : (!fir.array<32xi32>, i32) -> !fir.array<32xi32>
  fir.has_value %2 : !fir.array<32xi32>
}

// CHECK:          llvm.mlir.global internal @_QEmultiarray() : !llvm.array<32 x i32> {
// CHECK:            %[[CST:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:            %{{.*}} = llvm.mlir.undef : !llvm.array<32 x i32>
// CHECK:            %{{.*}} = llvm.insertvalue %[[CST]], %{{.*}}[5] : !llvm.array<32 x i32>
// CHECK-COUNT-24:   %{{.*}} = llvm.insertvalue %[[CST]], %{{.*}}[{{.*}}] : !llvm.array<32 x i32>
// CHECK:            %{{.*}} = llvm.insertvalue %[[CST]], %{{.*}}[31] : !llvm.array<32 x i32>
// CHECK-NOT:        llvm.insertvalue
// CHECK:            llvm.return %{{.*}} : !llvm.array<32 x i32>
// CHECK:          }

// -----

// Test fir.zero_bits operation with LLVM ptr type

func @zero_test_ptr() {
  %z = fir.zero_bits !llvm.ptr<f32>
  return
}

// CHECK: %{{.*}} = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NOT: fir.zero_bits

// -----

// Test fir.zero_bits operation with integer type.

func @zero_test_integer() {
  %z0 = fir.zero_bits i8
  %z1 = fir.zero_bits i16
  %z2 = fir.zero_bits i32
  %z3 = fir.zero_bits i64
  return
}

// CHECK: %{{.*}} = llvm.mlir.constant(0 : i8) : i8
// CHECK: %{{.*}} = llvm.mlir.constant(0 : i16) : i16
// CHECK: %{{.*}} = llvm.mlir.constant(0 : i32) : i32
// CHECK: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
// CHECK-NOT: fir.zero_bits

// -----

// Test fir.zero_bits operation with floating points types.

func @zero_test_float() {
  %z0 = fir.zero_bits f16
  %z1 = fir.zero_bits bf16
  %z2 = fir.zero_bits f32
  %z3 = fir.zero_bits f64
  %z4 = fir.zero_bits f80
  %z5 = fir.zero_bits f128
  return
}

// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f16) : f16
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f64) : f64
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f80) : f80
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f128) : f128
// CHECK-NOT: fir.zero_bits

// -----

// Verify that fir.unreachable is transformed to llvm.unreachable

// CHECK:  llvm.func @test_unreachable() {
// CHECK-NEXT:    llvm.unreachable
// CHECK-NEXT:  }

func @test_unreachable() {
  fir.unreachable
}
