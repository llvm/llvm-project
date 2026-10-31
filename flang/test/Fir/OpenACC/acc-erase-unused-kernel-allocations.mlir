// RUN: fir-opt %s --acc-erase-unused-kernel-allocations | FileCheck %s

// Storage that is never read or written is deleted, including an unused
// private recipe. A source array and a recipe allocation that are stored to
// remain.

module {
  func.func @erase_dead_keep_used() {
    %c1 = arith.constant 1 : index
    %pw = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%arg0 = %pw) {
      %n = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %shape = fir.shape %n : (index) -> !fir.shape<1>
      %dead = fir.allocmem !fir.array<?xf64>, %n {bindc_name = "dead", uniq_name = "dead"}
      %dead_ref = fir.convert %dead : (!fir.heap<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
      %dead_decl = fir.declare %dead_ref(%shape) uniq_name("dead") : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf64>>
      fir.freemem %dead : !fir.heap<!fir.array<?xf64>>
      %recipe_unused = fir.allocmem !fir.array<2xf32> {bindc_name = "acc.private.init", uniq_name = ""}
      %recipe_shape = fir.shape %c2 : (index) -> !fir.shape<1>
      %recipe_book = fir.allocmem !fir.array<4xf64> {bindc_name = "acc.private.init", uniq_name = ""}
      %recipe_book_decl = fir.declare %recipe_book(%recipe_shape) uniq_name("acc.private.init") : (!fir.heap<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.heap<!fir.array<4xf64>>
      fir.freemem %recipe_book_decl : !fir.heap<!fir.array<4xf64>>
      %live = fir.allocmem !fir.array<?xf64>, %n {bindc_name = "live", uniq_name = "live"}
      %live_ref = fir.convert %live : (!fir.heap<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
      %live_decl = fir.declare %live_ref(%shape) uniq_name("live") : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf64>>
      %elem = fir.array_coor %live_decl(%shape) %c0 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
      %val = fir.load %elem : !fir.ref<f64>
      fir.store %val to %elem : !fir.ref<f64>
      fir.freemem %live : !fir.heap<!fir.array<?xf64>>
      %recipe_live = fir.allocmem !fir.array<2xf32> {bindc_name = "acc.private.init", uniq_name = ""}
      %recipe_live_ref = fir.convert %recipe_live : (!fir.heap<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
      %recipe_live_decl = fir.declare %recipe_live_ref(%recipe_shape) uniq_name("acc.private.init") : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<2xf32>>
      %recipe_elem = fir.array_coor %recipe_live_decl(%recipe_shape) %c0 : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
      %one = arith.constant 1.0 : f32
      fir.store %one to %recipe_elem : !fir.ref<f32>
      acc.yield
    } <{origin = "acc.routine"}>
    return
  }

  // A dynamic stack or memref heap allocation that is never read or written
  // is deleted. A stack slot outside the compute region is left alone.
  func.func @erase_dead_stack_keep_used() {
    %c1 = arith.constant 1 : index
    %pw = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
    acc.compute_region launch(%arg0 = %pw) {
      %n = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %shape = fir.shape %n : (index) -> !fir.shape<1>
      %shift = fir.shape_shift %c0, %n : (index, index) -> !fir.shapeshift<1>
      %dead_stack = memref.alloca(%n) {bindc_name = "dead_stack", uniq_name = "dead_stack"} : memref<?xf64>
      %dead_stack_ref = fir.convert %dead_stack : (memref<?xf64>) -> !fir.ref<!fir.array<?xf64>>
      %dead_stack_decl = fir.declare %dead_stack_ref(%shift) uniq_name("dead_stack") : (!fir.ref<!fir.array<?xf64>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<?xf64>>
      %dead_alloca = fir.alloca !fir.array<?xf64>, %n {bindc_name = "dead_alloca", uniq_name = "dead_alloca"}
      %dead_alloca_decl = fir.declare %dead_alloca(%shape) uniq_name("dead_alloca") : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf64>>
      %dead_heap = memref.alloc(%n) {bindc_name = "dead_heap", uniq_name = "dead_heap"} : memref<?xf64>
      memref.dealloc %dead_heap : memref<?xf64>
      %live_stack = memref.alloca(%n) {bindc_name = "live_stack", uniq_name = "live_stack"} : memref<?xf64>
      %live_stack_ref = fir.convert %live_stack : (memref<?xf64>) -> !fir.ref<!fir.array<?xf64>>
      %live_stack_decl = fir.declare %live_stack_ref(%shape) uniq_name("live_stack") : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf64>>
      %elem = fir.array_coor %live_stack_decl(%shape) %c0 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
      %val = fir.load %elem : !fir.ref<f64>
      fir.store %val to %elem : !fir.ref<f64>
      %live_alloca = fir.alloca f64 {bindc_name = "live_alloca", uniq_name = "live_alloca"}
      %zero = arith.constant 0.0 : f64
      fir.store %zero to %live_alloca : !fir.ref<f64>
      acc.yield
    } <{origin = "acc.routine"}>
    %host_stack = memref.alloca() {uniq_name = "host_stack"} : memref<f64>
    return
  }
}

// CHECK-LABEL: func.func @erase_dead_keep_used
// CHECK-NOT: uniq_name = "dead"
// CHECK-NOT: fir.allocmem
// CHECK: fir.allocmem {{.*}}uniq_name = "live"
// CHECK-NOT: fir.allocmem
// CHECK: fir.allocmem {{.*}}bindc_name = "acc.private.init"
// CHECK-NOT: fir.allocmem
// CHECK-NOT: uniq_name = "dead"

// CHECK-LABEL: func.func @erase_dead_stack_keep_used
// CHECK-NOT: uniq_name = "dead_stack"
// CHECK-NOT: uniq_name = "dead_alloca"
// CHECK-NOT: uniq_name = "dead_heap"
// CHECK: memref.alloca{{.*}}uniq_name = "live_stack"
// CHECK: fir.alloca {{.*}}uniq_name = "live_alloca"
// CHECK: uniq_name = "host_stack"
// CHECK-NOT: uniq_name = "dead_stack"
// CHECK-NOT: uniq_name = "dead_alloca"
// CHECK-NOT: uniq_name = "dead_heap"
