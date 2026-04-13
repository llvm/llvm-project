// RUN: fir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=store}))" 2>&1 | FileCheck %s

func.func @test_store_scalar_f32() {
  %ptr = fir.alloca f32 {test.ptr}
  // CHECK: Successfully generated store for operation: %{{.*}} = fir.alloca f32 {test.ptr}
  // CHECK: Generated: %[[VAL:.*]] = arith.constant 4.200000e+01 : f32
  // CHECK: Generated: fir.store %[[VAL]] to %{{.*}} : !fir.ref<f32>
  return
}

// -----

func.func @test_store_scalar_i32() {
  %ptr = fir.alloca i32 {test.ptr}
  // CHECK: Successfully generated store for operation: %{{.*}} = fir.alloca i32 {test.ptr}
  // CHECK: Generated: %[[VAL:.*]] = arith.constant 42 : i32
  // CHECK: Generated: fir.store %[[VAL]] to %{{.*}} : !fir.ref<i32>
  return
}

// -----

func.func @test_store_scalar_i64() {
  %ptr = fir.alloca i64 {test.ptr}
  // CHECK: Successfully generated store for operation: %{{.*}} = fir.alloca i64 {test.ptr}
  // CHECK: Generated: %[[VAL:.*]] = arith.constant 42 : i64
  // CHECK: Generated: fir.store %[[VAL]] to %{{.*}} : !fir.ref<i64>
  return
}

// -----

func.func @test_store_heap_scalar() {
  %ptr = fir.allocmem f64 {test.ptr}
  // CHECK: Successfully generated store for operation: %{{.*}} = fir.allocmem f64 {test.ptr}
  // CHECK: Generated: %[[VAL:.*]] = arith.constant 4.200000e+01 : f64
  // CHECK: Generated: fir.store %[[VAL]] to %{{.*}} : !fir.heap<f64>
  return
}

// -----

func.func @test_store_with_type_conversion() {
  %ptr = fir.alloca i32 {test.ptr}
  // CHECK: Successfully generated store for operation: %{{.*}} = fir.alloca i32 {test.ptr}
  // CHECK: Generated: %[[VAL:.*]] = arith.constant 42 : i32
  // CHECK: Generated: fir.store %[[VAL]] to %{{.*}} : !fir.ref<i32>
  return
}

// -----

func.func @test_store_constant_array() {
  %val = fir.undefined !fir.array<10xf32> {test.value}
  %ptr = fir.alloca !fir.array<10xf32> {test.ptr}
  // CHECK: Successfully generated store for operation: %{{.*}} = fir.alloca !fir.array<10xf32> {test.ptr}
  // CHECK: Generated: fir.store %{{.*}} to %{{.*}} : !fir.ref<!fir.array<10xf32>>
  return
}

// -----

func.func @test_store_dynamic_array_fails() {
  %c10 = arith.constant 10 : index
  %ptr = fir.alloca !fir.array<?xf32>, %c10 {test.ptr}
  // CHECK: Failed to generate store for operation: %{{.*}} = fir.alloca !fir.array<?xf32>
  return
}

// -----

func.func @test_store_box_fails() {
  %ptr = fir.alloca !fir.box<!fir.ptr<f32>> {test.ptr}
  // CHECK: Failed to generate store for operation: %{{.*}} = fir.alloca !fir.box<!fir.ptr<f32>>
  return
}

// -----

func.func @test_store_unlimited_polymorphic_fails() {
  %ptr = fir.alloca !fir.class<none> {test.ptr}
  // CHECK: Failed to generate store for operation: %{{.*}} = fir.alloca !fir.class<none>
  return
}

