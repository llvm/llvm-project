// RUN: fir-opt %s --split-input-file --pass-pipeline="builtin.module(test-acc-recipe-populate{recipe-type=private})" | FileCheck %s

// The tests here use a synthetic hlfir.declare in order to ensure that the hlfir dialect is
// loaded. This is required because the pass used is part of OpenACC test passes outside of
// flang and the APIs being test may generate hlfir even when it does not appear.

// Test scalar type (f32)
// CHECK: acc.private.recipe @private_scalar : !fir.ref<f32> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca f32
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "scalar"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<f32>
// CHECK: }
// CHECK-NOT: destroy

func.func @test_scalar() {
  %0 = fir.alloca f32 {test.var = "scalar"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test logical type
// CHECK: acc.private.recipe @private_logical : !fir.ref<!fir.logical<4>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.logical<4>>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca !fir.logical<4>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "logical"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.logical<4>>
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
// CHECK: acc.private.recipe @private_complex : !fir.ref<complex<f32>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<complex<f32>>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca complex<f32>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "complex"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<complex<f32>>
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
// CHECK: acc.private.recipe @private_array_1d : !fir.ref<!fir.array<100xf32>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
// CHECK:   %[[C100:.*]] = arith.constant 100 : index
// CHECK:   %[[SHAPE:.*]] = fir.shape %[[C100]] : (index) -> !fir.shape<1>
// CHECK:   %[[ALLOC:.*]] = fir.alloca !fir.array<100xf32>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = "array_1d"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.array<100xf32>>
// CHECK: }
// CHECK-NOT: destroy

func.func @test_array_1d() {
  %0 = fir.alloca !fir.array<100xf32> {test.var = "array_1d"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test 3D static array
// CHECK: acc.private.recipe @private_array_3d : !fir.ref<!fir.array<5x10x15xi32>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<5x10x15xi32>>):
// CHECK:   %[[C5:.*]] = arith.constant 5 : index
// CHECK:   %[[C10:.*]] = arith.constant 10 : index
// CHECK:   %[[C15:.*]] = arith.constant 15 : index
// CHECK:   %[[SHAPE:.*]] = fir.shape %[[C5]], %[[C10]], %[[C15]] : (index, index, index) -> !fir.shape<3>
// CHECK:   %[[ALLOC:.*]] = fir.alloca !fir.array<5x10x15xi32>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = "array_3d"} : (!fir.ref<!fir.array<5x10x15xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<5x10x15xi32>>, !fir.ref<!fir.array<5x10x15xi32>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.array<5x10x15xi32>>
// CHECK: }
// CHECK-NOT: destroy

func.func @test_array_3d() {
  %0 = fir.alloca !fir.array<5x10x15xi32> {test.var = "array_3d"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test derived type with multiple fields
// CHECK: acc.private.recipe @private_derived : !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>):
// CHECK:   %[[ALLOC:.*]] = fir.alloca !fir.type<_QTpoint{x:f32,y:f32,z:f32}>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[ALLOC]] {uniq_name = "derived"} : (!fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>) -> (!fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>, !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.type<_QTpoint{x:f32,y:f32,z:f32}>>
// CHECK: }
// CHECK-NOT: destroy

func.func @test_derived() {
  %0 = fir.alloca !fir.type<_QTpoint{x:f32,y:f32,z:f32}> {test.var = "derived"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test box type with heap scalar (needs destroy)
// CHECK: acc.private.recipe @private_box_heap_scalar : !fir.ref<!fir.box<!fir.heap<f64>>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<f64>>>):
// CHECK:   %[[BOXALLOC:.*]] = fir.alloca !fir.box<!fir.heap<f64>>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[BOXALLOC]] {uniq_name = "box_heap_scalar"} : (!fir.ref<!fir.box<!fir.heap<f64>>>) -> (!fir.ref<!fir.box<!fir.heap<f64>>>, !fir.ref<!fir.box<!fir.heap<f64>>>)
// CHECK:   %[[SCALAR:.*]] = fir.allocmem f64
// CHECK:   %[[EMBOX:.*]] = fir.embox %[[SCALAR]] : (!fir.heap<f64>) -> !fir.box<!fir.heap<f64>>
// CHECK:   fir.store %[[EMBOX]] to %{{.*}}#0 : !fir.ref<!fir.box<!fir.heap<f64>>>
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.box<!fir.heap<f64>>>
// CHECK: } destroy {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<f64>>>, %{{.*}}: !fir.ref<!fir.box<!fir.heap<f64>>>):
// CHECK:   acc.terminator
// CHECK: }

func.func @test_box_heap_scalar() {
  %0 = fir.alloca !fir.box<!fir.heap<f64>> {test.var = "box_heap_scalar"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test box type with pointer scalar (needs destroy)
// CHECK: acc.private.recipe @private_box_ptr_scalar : !fir.ref<!fir.box<!fir.ptr<i32>>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>>):
// CHECK:   %[[BOXALLOC:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[BOXALLOC]] {uniq_name = "box_ptr_scalar"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
// CHECK:   %[[SCALAR:.*]] = fir.allocmem i32
// CHECK:   %[[EMBOX:.*]] = fir.embox %[[SCALAR]] : (!fir.heap<i32>) -> !fir.box<!fir.ptr<i32>>
// CHECK:   fir.store %[[EMBOX]] to %{{.*}}#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK: } destroy {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>>, %{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>>):
// CHECK:   acc.terminator
// CHECK: }

func.func @test_box_ptr_scalar() {
  %0 = fir.alloca !fir.box<!fir.ptr<i32>> {test.var = "box_ptr_scalar"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test box type with 1D heap array (needs destroy)
// CHECK: acc.private.recipe @private_box_heap_array_1d : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>):
// CHECK:   %[[BOXALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[BOXALLOC]] {uniq_name = "box_heap_array_1d"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK: } destroy {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>):
// CHECK:   acc.terminator
// CHECK: }

func.func @test_box_heap_array_1d() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {test.var = "box_heap_array_1d"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test box type with 2D heap array (needs destroy)
// CHECK: acc.private.recipe @private_box_heap_array_2d : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi64>>>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi64>>>>):
// CHECK:   %[[BOXALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi64>>>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[BOXALLOC]] {uniq_name = "box_heap_array_2d"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi64>>>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi64>>>>
// CHECK: } destroy {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi64>>>>, %{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi64>>>>):
// CHECK:   acc.terminator
// CHECK: }

func.func @test_box_heap_array_2d() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi64>>> {test.var = "box_heap_array_2d"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}

// -----

// Test box type with pointer array (needs destroy)
// CHECK: acc.private.recipe @private_box_ptr_array : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> init {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>):
// CHECK:   %[[BOXALLOC:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
// CHECK:   %{{.*}}:2 = hlfir.declare %[[BOXALLOC]] {uniq_name = "box_ptr_array"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
// CHECK:   acc.yield %{{.*}}#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK: } destroy {
// CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>):
// CHECK:   acc.terminator
// CHECK: }

func.func @test_box_ptr_array() {
  %0 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {test.var = "box_ptr_array"}
  %var = fir.alloca f32
  %1:2 = hlfir.declare %var {uniq_name = "load_hlfir"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}
