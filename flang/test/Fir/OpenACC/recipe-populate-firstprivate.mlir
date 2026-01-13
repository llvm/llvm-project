// RUN: fir-opt %s --split-input-file --pass-pipeline="builtin.module(test-acc-recipe-populate{recipe-type=firstprivate})" | FileCheck %s

// The tests here use a synthetic hlfir.declare in order to ensure that the hlfir dialect is
// loaded. This is required because the pass used is part of OpenACC test passes outside of
// flang and the APIs being test may generate hlfir even when it does not appear.

// Test scalar type (f32)
// CHECK: acc.firstprivate.recipe @firstprivate_scalar : !fir.ref<f32> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca f32
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "scalar"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<f32>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<f32>, %[[DST:.*]]: !fir.ref<f32>):
// CHECK:   %[[LOAD:.*]] = fir.load %[[SRC]] : !fir.ref<f32>
// CHECK:   fir.store %[[LOAD]] to %[[DST]] : !fir.ref<f32>
// CHECK:   acc.terminator
// CHECK: }
// CHECK-NOT: destroy

func.func @test_scalar() {
  %0 = fir.alloca f32 {test.var = "scalar"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test integer scalar
// CHECK: acc.firstprivate.recipe @firstprivate_int : !fir.ref<i32> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca i32
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "int"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<i32>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<i32>, %[[DST:.*]]: !fir.ref<i32>):
// CHECK:   %[[LOAD:.*]] = fir.load %[[SRC]] : !fir.ref<i32>
// CHECK:   fir.store %[[LOAD]] to %[[DST]] : !fir.ref<i32>
// CHECK:   acc.terminator
// CHECK: }
// CHECK-NOT: destroy

func.func @test_int() {
  %0 = fir.alloca i32 {test.var = "int"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test logical type
// CHECK: acc.firstprivate.recipe @firstprivate_logical : !fir.ref<!fir.logical<4>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.logical<4>>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca !fir.logical<4>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "logical"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.logical<4>>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.logical<4>>, %[[DST:.*]]: !fir.ref<!fir.logical<4>>):
// CHECK:   %[[LOAD:.*]] = fir.load %[[SRC]] : !fir.ref<!fir.logical<4>>
// CHECK:   fir.store %[[LOAD]] to %[[DST]] : !fir.ref<!fir.logical<4>>
// CHECK:   acc.terminator
// CHECK: }
// CHECK-NOT: destroy

func.func @test_logical() {
  %0 = fir.alloca !fir.logical<4> {test.var = "logical"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test complex type
// CHECK: acc.firstprivate.recipe @firstprivate_complex : !fir.ref<complex<f32>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<complex<f32>>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca complex<f32>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "complex"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<complex<f32>>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<complex<f32>>, %[[DST:.*]]: !fir.ref<complex<f32>>):
// CHECK:   %[[LOAD:.*]] = fir.load %[[SRC]] : !fir.ref<complex<f32>>
// CHECK:   fir.store %[[LOAD]] to %[[DST]] : !fir.ref<complex<f32>>
// CHECK:   acc.terminator
// CHECK: }
// CHECK-NOT: destroy

func.func @test_complex() {
  %0 = fir.alloca complex<f32> {test.var = "complex"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test 1D static array
// CHECK: acc.firstprivate.recipe @firstprivate_array_1d : !fir.ref<!fir.array<100xf32>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
// CHECK:   %[[C100:.*]] = arith.constant 100 : index
// CHECK:   %[[SHAPE:.*]] = fir.shape %[[C100]] : (index) -> !fir.shape<1>
// CHECK:   %[[ALLOC:.*]] = fir.alloca !fir.array<100xf32>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = "array_1d"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.array<100xf32>>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.array<100xf32>>, %[[DST:.*]]: !fir.ref<!fir.array<100xf32>>):
// CHECK:   hlfir.assign %[[SRC]] to %[[DST]] : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>
// CHECK:   acc.terminator
// CHECK: }
// CHECK-NOT: destroy

func.func @test_array_1d() {
  %0 = fir.alloca !fir.array<100xf32> {test.var = "array_1d"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test 2D static array
// CHECK: acc.firstprivate.recipe @firstprivate_array_2d : !fir.ref<!fir.array<10x20xi32>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<10x20xi32>>):
// CHECK:   %[[C10:.*]] = arith.constant 10 : index
// CHECK:   %[[C20:.*]] = arith.constant 20 : index
// CHECK:   %[[SHAPE:.*]] = fir.shape %[[C10]], %[[C20]] : (index, index) -> !fir.shape<2>
// CHECK:   %[[ALLOC:.*]] = fir.alloca !fir.array<10x20xi32>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = "array_2d"} : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x20xi32>>, !fir.ref<!fir.array<10x20xi32>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.array<10x20xi32>>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.array<10x20xi32>>, %[[DST:.*]]: !fir.ref<!fir.array<10x20xi32>>):
// CHECK:   hlfir.assign %[[SRC]] to %[[DST]] : !fir.ref<!fir.array<10x20xi32>>, !fir.ref<!fir.array<10x20xi32>>
// CHECK:   acc.terminator
// CHECK: }
// CHECK-NOT: destroy

func.func @test_array_2d() {
  %0 = fir.alloca !fir.array<10x20xi32> {test.var = "array_2d"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test derived type with multiple fields
// CHECK: acc.firstprivate.recipe @firstprivate_derived : !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca !fir.type<_QTpoint{x:f32,y:f32,z:f32}>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "derived"} : (!fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>) -> (!fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>
// CHECK: } copy {
// CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>, %[[DST:.*]]: !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>):
// CHECK:   hlfir.assign %[[SRC]] to %[[DST]] : !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>
// CHECK:   acc.terminator
// CHECK: }
// CHECK-NOT: destroy

func.func @test_derived() {
  %0 = fir.alloca !fir.type<_QTpoint{x:f32,y:f32,z:f32}> {test.var = "derived"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}
