// RUN: fir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=alloc}))" 2>&1 | FileCheck %s

// The tests here use a synthetic hlfir.declare in order to ensure that the hlfir dialect is
// loaded. This is required because the pass used is part of OpenACC test passes outside of
// flang and the APIs being test may generate hlfir even when it does not appear.

func.func @test_ref_scalar_alloc() {
  %0 = fir.alloca f32 {test.ptr}
  %1:2 = hlfir.declare %0 {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated alloc for operation: %{{.*}} = fir.alloca f32 {test.ptr}
  // CHECK: Generated: %{{.*}} = fir.alloca f32
  return
}

// -----

func.func @test_ref_static_array_alloc() {
  %0 = fir.alloca !fir.array<10x20xf32> {test.ptr}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated alloc for operation: %{{.*}} = fir.alloca !fir.array<10x20xf32> {test.ptr}
  // CHECK: Generated: %{{.*}} = fir.alloca !fir.array<10x20xf32>
  return
}

// -----

func.func @test_ref_derived_type_alloc() {
  %0 = fir.alloca !fir.type<_QTt{i:i32}> {test.ptr}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated alloc for operation: %{{.*}} = fir.alloca !fir.type<_QTt{i:i32}> {test.ptr}
  // CHECK: Generated: %{{.*}} = fir.alloca !fir.type<_QTt{i:i32}>
  return
}

// -----

func.func @test_heap_scalar_alloc() {
  %0 = fir.allocmem f32 {test.ptr}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated alloc for operation: %{{.*}} = fir.allocmem f32 {test.ptr}
  // CHECK: Generated: %{{.*}} = fir.allocmem f32
  return
}

// -----

func.func @test_heap_static_array_alloc() {
  %0 = fir.allocmem !fir.array<10x20xf32> {test.ptr}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated alloc for operation: %{{.*}} = fir.allocmem !fir.array<10x20xf32> {test.ptr}
  // CHECK: Generated: %{{.*}} = fir.allocmem !fir.array<10x20xf32>
  return
}

// -----

func.func @test_ptr_scalar_alloc() {
  %0 = fir.alloca f32
  %1 = fir.convert %0 {test.ptr} : (!fir.ref<f32>) -> !fir.ptr<f32>
  %2:2 = hlfir.declare %0 {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated alloc for operation
  // CHECK: Generated: %{{.*}} = fir.alloca f32
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.ptr<f32>
  return
}

// -----

func.func @test_llvm_ptr_scalar_alloc() {
  %0 = fir.alloca f32
  %1 = fir.convert %0 {test.ptr} : (!fir.ref<f32>) -> !fir.llvm_ptr<f32>
  %2:2 = hlfir.declare %0 {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated alloc for operation
  // CHECK: Generated: %{{.*}} = fir.alloca f32
  // CHECK: Generated: %{{.*}} = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.llvm_ptr<f32>
  return
}

// -----

func.func @test_dynamic_array_alloc_fails(%arg0: !fir.ref<!fir.array<?xf32>>) {
  %0 = fir.convert %arg0 {test.ptr} : (!fir.ref<!fir.array<?xf32>>) -> !fir.llvm_ptr<!fir.array<?xf32>>
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Failed to generate alloc for operation: %{{.*}} = fir.convert %{{.*}} {test.ptr} : (!fir.ref<!fir.array<?xf32>>) -> !fir.llvm_ptr<!fir.array<?xf32>>
  return
}

// -----

func.func @test_unlimited_polymorphic_alloc_fails() {
  %0 = fir.alloca !fir.class<none> {test.ptr}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Failed to generate alloc for operation: %{{.*}} = fir.alloca !fir.class<none> {test.ptr}
  return
}

// -----

func.func @test_dynamic_char_alloc_fails(%arg0: !fir.ref<!fir.char<1,?>>) {
  %0 = fir.convert %arg0 {test.ptr} : (!fir.ref<!fir.char<1,?>>) -> !fir.llvm_ptr<!fir.char<1,?>>
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Failed to generate alloc for operation: %{{.*}} = fir.convert %{{.*}} {test.ptr} : (!fir.ref<!fir.char<1,?>>) -> !fir.llvm_ptr<!fir.char<1,?>>
  return
}

// -----

func.func @test_static_char_alloc() {
  %0 = fir.alloca !fir.char<1,10> {test.ptr}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated alloc for operation: %{{.*}} = fir.alloca !fir.char<1,10> {test.ptr}
  // CHECK: Generated: %{{.*}} = fir.alloca !fir.char<1,10>
  return
}
