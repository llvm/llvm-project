// RUN: fir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=cast}))" 2>&1 | FileCheck %s

func.func @test_fir_ref_to_memref_scalar() {
  %0 = fir.alloca f32 {test.cast, cast_dest = memref<f32>}
  // CHECK: Successfully generated cast for operation: %{{.*}} = fir.alloca f32{{.*}}
  // CHECK: Cast result type: memref<f32>
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<f32>) -> memref<f32>
  return
}

// -----

func.func @test_memref_to_fir_ref_scalar() {
  %0 = memref.alloca() {test.cast, cast_dest = !fir.ref<f32>} : memref<f32>
  // CHECK: Successfully generated cast for operation: %{{.*}} = memref.alloca(){{.*}}
  // CHECK: Cast result type: !fir.ref<f32>
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (memref<f32>) -> !fir.ref<f32>
  return
}

// -----

func.func @test_fir_ref_identity() {
  %0 = fir.alloca i32 {test.cast, cast_dest = !fir.ref<i32>}
  // CHECK: Successfully generated cast for operation: %{{.*}} = fir.alloca i32{{.*}}
  // CHECK: Cast result type: !fir.ref<i32>
  return
}

// -----

func.func @test_i64_to_fir_ref() {
  %0 = arith.constant {test.cast, cast_dest = !fir.ref<i8>} 0 : i64
  // CHECK: Successfully generated cast for operation: %{{.*}} = arith.constant{{.*}}
  // CHECK: Cast result type: !fir.ref<i8>
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (i64) -> !fir.ref<i8>
  return
}

// -----

func.func @test_index_to_fir_ptr() {
  %0 = arith.constant {test.cast, cast_dest = !fir.ptr<i8>} 0 : index
  // CHECK: Successfully generated cast for operation: %{{.*}} = arith.constant{{.*}}
  // CHECK: Cast result type: !fir.ptr<i8>
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (index) -> !fir.ptr<i8>
  return
}

// -----

func.func @test_fir_heap_to_i64() {
  %0 = fir.zero_bits !fir.heap<i8> {test.cast, cast_dest = i64}
  // CHECK: Successfully generated cast for operation: %{{.*}} = fir.zero_bits{{.*}}
  // CHECK: Cast result type: i64
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (!fir.heap<i8>) -> i64
  return
}

// -----

func.func @test_fir_llvm_ptr_to_index() {
  %0 = fir.zero_bits !fir.llvm_ptr<i8> {test.cast, cast_dest = index}
  // CHECK: Successfully generated cast for operation: %{{.*}} = fir.zero_bits{{.*}}
  // CHECK: Cast result type: index
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (!fir.llvm_ptr<i8>) -> index
  return
}
