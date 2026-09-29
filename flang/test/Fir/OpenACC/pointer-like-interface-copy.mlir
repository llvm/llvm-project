// RUN: fir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=copy}))" 2>&1 | FileCheck %s

// The tests here use a synthetic hlfir.declare in order to ensure that the hlfir dialect is
// loaded. This is required because the pass used is part of OpenACC test passes outside of
// flang and the APIs being test may generate hlfir even when it does not appear.

func.func @test_copy_scalar() {
  %src = fir.alloca f32 {test.src_ptr}
  %dest = fir.alloca f32 {test.dest_ptr}
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated copy from source: %{{.*}} = fir.alloca f32 {test.src_ptr} to destination: %{{.*}} = fir.alloca f32 {test.dest_ptr}
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.ref<f32>
  // CHECK: Generated: fir.store %{{.*}} to %{{.*}} : !fir.ref<f32>
  return
}

// -----

func.func @test_copy_static_array() {
  %src = fir.alloca !fir.array<10x20xf32> {test.src_ptr}
  %dest = fir.alloca !fir.array<10x20xf32> {test.dest_ptr}
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated copy from source: %{{.*}} = fir.alloca !fir.array<10x20xf32> {test.src_ptr} to destination: %{{.*}} = fir.alloca !fir.array<10x20xf32> {test.dest_ptr}
  // CHECK: Generated: hlfir.assign %{{.*}} to %{{.*}} : !fir.ref<!fir.array<10x20xf32>>, !fir.ref<!fir.array<10x20xf32>>
  return
}

// -----

func.func @test_copy_derived_type() {
  %src = fir.alloca !fir.type<_QTt{i:i32}> {test.src_ptr}
  %dest = fir.alloca !fir.type<_QTt{i:i32}> {test.dest_ptr}
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated copy from source: %{{.*}} = fir.alloca !fir.type<_QTt{i:i32}> {test.src_ptr} to destination: %{{.*}} = fir.alloca !fir.type<_QTt{i:i32}> {test.dest_ptr}
  // CHECK: Generated: hlfir.assign %{{.*}} to %{{.*}} : !fir.ref<!fir.type<_QTt{i:i32}>>, !fir.ref<!fir.type<_QTt{i:i32}>>
  return
}

// -----

func.func @test_copy_heap_scalar() {
  %src = fir.allocmem f32 {test.src_ptr}
  %dest = fir.allocmem f32 {test.dest_ptr}
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated copy from source: %{{.*}} = fir.allocmem f32 {test.src_ptr} to destination: %{{.*}} = fir.allocmem f32 {test.dest_ptr}
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.heap<f32>
  // CHECK: Generated: fir.store %{{.*}} to %{{.*}} : !fir.heap<f32>
  return
}

// -----

func.func @test_copy_static_char() {
  %src = fir.alloca !fir.char<1,10> {test.src_ptr}
  %dest = fir.alloca !fir.char<1,10> {test.dest_ptr}
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Successfully generated copy from source: %{{.*}} = fir.alloca !fir.char<1,10> {test.src_ptr} to destination: %{{.*}} = fir.alloca !fir.char<1,10> {test.dest_ptr}
  // CHECK: Generated: hlfir.assign %{{.*}} to %{{.*}} : !fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>
  return
}

// -----

func.func @test_copy_mismatched_types_fails() {
  %src = fir.alloca f32 {test.src_ptr}
  %dest = fir.alloca f64 {test.dest_ptr}
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Failed to generate copy from source: %{{.*}} = fir.alloca f32 {test.src_ptr} to destination: %{{.*}} = fir.alloca f64 {test.dest_ptr}
  return
}

// -----

func.func @test_copy_mismatched_shapes_fails() {
  %src = fir.alloca !fir.array<10xf32> {test.src_ptr}
  %dest = fir.alloca !fir.array<20xf32> {test.dest_ptr}
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Failed to generate copy from source: %{{.*}} = fir.alloca !fir.array<10xf32> {test.src_ptr} to destination: %{{.*}} = fir.alloca !fir.array<20xf32> {test.dest_ptr}
  return
}

// -----

func.func @test_copy_dynamic_array_fails(%arg0: !fir.ref<!fir.array<?xf32>>, %arg1: !fir.ref<!fir.array<?xf32>>) {
  %src = fir.convert %arg0 {test.src_ptr} : (!fir.ref<!fir.array<?xf32>>) -> !fir.llvm_ptr<!fir.array<?xf32>>
  %dest = fir.convert %arg1 {test.dest_ptr} : (!fir.ref<!fir.array<?xf32>>) -> !fir.llvm_ptr<!fir.array<?xf32>>
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Failed to generate copy from source: %{{.*}} = fir.convert %{{.*}} {test.src_ptr} : (!fir.ref<!fir.array<?xf32>>) -> !fir.llvm_ptr<!fir.array<?xf32>> to destination: %{{.*}} = fir.convert %{{.*}} {test.dest_ptr} : (!fir.ref<!fir.array<?xf32>>) -> !fir.llvm_ptr<!fir.array<?xf32>>
  return
}

// -----

func.func @test_copy_unlimited_polymorphic_fails() {
  %src = fir.alloca !fir.class<none> {test.src_ptr}
  %dest = fir.alloca !fir.class<none> {test.dest_ptr}
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Failed to generate copy from source: %{{.*}} = fir.alloca !fir.class<none> {test.src_ptr} to destination: %{{.*}} = fir.alloca !fir.class<none> {test.dest_ptr}
  return
}

// -----

func.func @test_copy_dynamic_char_fails(%arg0: !fir.ref<!fir.char<1,?>>, %arg1: !fir.ref<!fir.char<1,?>>) {
  %src = fir.convert %arg0 {test.src_ptr} : (!fir.ref<!fir.char<1,?>>) -> !fir.llvm_ptr<!fir.char<1,?>>
  %dest = fir.convert %arg1 {test.dest_ptr} : (!fir.ref<!fir.char<1,?>>) -> !fir.llvm_ptr<!fir.char<1,?>>
  %var = fir.alloca f32
  %0:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  // CHECK: Failed to generate copy from source: %{{.*}} = fir.convert %{{.*}} {test.src_ptr} : (!fir.ref<!fir.char<1,?>>) -> !fir.llvm_ptr<!fir.char<1,?>> to destination: %{{.*}} = fir.convert %{{.*}} {test.dest_ptr} : (!fir.ref<!fir.char<1,?>>) -> !fir.llvm_ptr<!fir.char<1,?>>
  return
}
