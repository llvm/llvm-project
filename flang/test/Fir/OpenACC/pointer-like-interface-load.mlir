// RUN: fir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=load}))" 2>&1 | FileCheck %s

func.func @test_load_scalar_f32() {
  %ptr = fir.alloca f32 {test.ptr}
  // CHECK: Successfully generated load for operation: %{{.*}} = fir.alloca f32 {test.ptr}
  // CHECK: Loaded value type: f32
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.ref<f32>
  return
}

// -----

func.func @test_load_scalar_i32() {
  %ptr = fir.alloca i32 {test.ptr}
  // CHECK: Successfully generated load for operation: %{{.*}} = fir.alloca i32 {test.ptr}
  // CHECK: Loaded value type: i32
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
  return
}

// -----

func.func @test_load_scalar_i64() {
  %ptr = fir.alloca i64 {test.ptr}
  // CHECK: Successfully generated load for operation: %{{.*}} = fir.alloca i64 {test.ptr}
  // CHECK: Loaded value type: i64
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.ref<i64>
  return
}

// -----

func.func @test_load_heap_scalar() {
  %ptr = fir.allocmem f64 {test.ptr}
  // CHECK: Successfully generated load for operation: %{{.*}} = fir.allocmem f64 {test.ptr}
  // CHECK: Loaded value type: f64
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.heap<f64>
  return
}

// -----

func.func @test_load_logical() {
  %ptr = fir.alloca !fir.logical<4> {test.ptr}
  // CHECK: Successfully generated load for operation: %{{.*}} = fir.alloca !fir.logical<4> {test.ptr}
  // CHECK: Loaded value type: !fir.logical<4>
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
  return
}

// -----

func.func @test_load_derived_type() {
  %ptr = fir.alloca !fir.type<_QTt{i:i32}> {test.ptr}
  // CHECK: Successfully generated load for operation: %{{.*}} = fir.alloca !fir.type<_QTt{i:i32}> {test.ptr}
  // CHECK: Loaded value type: !fir.type<_QTt{i:i32}>
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.ref<!fir.type<_QTt{i:i32}>>
  return
}

// -----

func.func @test_load_constant_array() {
  %ptr = fir.alloca !fir.array<10xf32> {test.ptr}
  // CHECK: Successfully generated load for operation: %{{.*}} = fir.alloca !fir.array<10xf32> {test.ptr}
  // CHECK: Loaded value type: !fir.array<10xf32>
  // CHECK: Generated: %{{.*}} = fir.load %{{.*}} : !fir.ref<!fir.array<10xf32>>
  return
}

// -----

func.func @test_load_dynamic_array_fails() {
  %c10 = arith.constant 10 : index
  %ptr = fir.alloca !fir.array<?xf32>, %c10 {test.ptr}
  // CHECK: Failed to generate load for operation: %{{.*}} = fir.alloca !fir.array<?xf32>
  return
}

// -----

func.func @test_load_box_fails() {
  %ptr = fir.alloca !fir.box<!fir.ptr<f32>> {test.ptr}
  // CHECK: Failed to generate load for operation: %{{.*}} = fir.alloca !fir.box<!fir.ptr<f32>>
  return
}

// -----

func.func @test_load_unlimited_polymorphic_fails() {
  %ptr = fir.alloca !fir.class<none> {test.ptr}
  // CHECK: Failed to generate load for operation: %{{.*}} = fir.alloca !fir.class<none>
  return
}

