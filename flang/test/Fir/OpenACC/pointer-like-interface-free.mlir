// RUN: fir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=free}))" 2>&1 | FileCheck %s

// The tests here use a synthetic hlfir.declare in order to ensure that the hlfir dialect is
// loaded. This is required because the pass used is part of OpenACC test passes outside of
// flang and the APIs being test may generate hlfir even when it does not appear.

func.func @test_ref_scalar_free() {
  %0 = fir.alloca f32 {test.ptr}
  %1:2 = hlfir.declare %0 {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated free for operation: %{{.*}} = fir.alloca f32 {test.ptr}
  // CHECK-NOT: Generated
  return
}

// -----

func.func @test_heap_scalar_free() {
  %0 = fir.allocmem f32 {test.ptr}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated free for operation: %{{.*}} = fir.allocmem f32 {test.ptr}
  // CHECK: Generated: fir.freemem %{{.*}} : !fir.heap<f32>
  return
}

// -----

func.func @test_heap_array_free() {
  %0 = fir.allocmem !fir.array<10x20xf32> {test.ptr}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated free for operation: %{{.*}} = fir.allocmem !fir.array<10x20xf32> {test.ptr}
  // CHECK: Generated: fir.freemem %{{.*}} : !fir.heap<!fir.array<10x20xf32>>
  return
}

// -----

func.func @test_convert_walking_free() {
  %0 = fir.alloca f32
  %1 = fir.convert %0 {test.ptr} : (!fir.ref<f32>) -> !fir.ptr<f32>
  %2:2 = hlfir.declare %0 {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated free for operation: %{{.*}} = fir.convert %{{.*}} {test.ptr} : (!fir.ref<f32>) -> !fir.ptr<f32>
  // CHECK-NOT: Generated
  return
}

// -----

func.func @test_declare_walking_free() {
  %0 = fir.alloca f32
  %1 = fir.declare %0 {test.ptr, uniq_name = "x"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %2:2 = hlfir.declare %0 {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated free for operation: %{{.*}} = fir.declare %{{.*}} {test.ptr, uniq_name = "x"} : (!fir.ref<f32>) -> !fir.ref<f32>
  // CHECK-NOT: Generated
  return
}

// -----

func.func @test_hlfir_declare_walking_free() {
  %0 = fir.alloca f32
  %1:2 = hlfir.declare %0 {test.ptr, uniq_name = "x"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %var = fir.alloca f32
  %2:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated free for operation
  // CHECK-NOT: Generated
  return
}

// -----

func.func @test_heap_through_convert_free() {
  %0 = fir.allocmem f32
  %1 = fir.convert %0 {test.ptr} : (!fir.heap<f32>) -> !fir.llvm_ptr<f32>
  %var = fir.alloca f32
  %2:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated free for operation: %{{.*}} = fir.convert %{{.*}} {test.ptr} : (!fir.heap<f32>) -> !fir.llvm_ptr<f32>
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (!fir.llvm_ptr<f32>) -> !fir.heap<f32>
  // CHECK: Generated: fir.freemem %{{.*}} : !fir.heap<f32>
  return
}

// -----

func.func @test_heap_through_declare_free() {
  %0 = fir.allocmem f32
  %1 = fir.declare %0 {test.ptr, uniq_name = "x"} : (!fir.heap<f32>) -> !fir.heap<f32>
  %var = fir.alloca f32
  %2:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated free for operation: %{{.*}} = fir.declare %{{.*}} {test.ptr, uniq_name = "x"} : (!fir.heap<f32>) -> !fir.heap<f32>
  // CHECK: Generated: fir.freemem %{{.*}} : !fir.heap<f32>
  return
}
