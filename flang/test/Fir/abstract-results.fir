// Test rewrite of functions that return fir.array<>, fir.type<>, fir.box<> to
// functions that take an additional argument for the result.

// RUN: fir-opt %s --abstract-result-on-func-opt | FileCheck %s
// RUN: fir-opt %s --abstract-result-on-func-opt=abstract-result-as-box | FileCheck %s --check-prefix=CHECK-BOX

// ----------------------- Test declaration rewrite ----------------------------

// CHECK-LABEL:  func private @arrayfunc(!fir.ref<!fir.array<?xf32>>, i32)
// CHECK-BOX-LABEL:  func private @arrayfunc(!fir.box<!fir.array<?xf32>>, i32)
func.func private @arrayfunc(i32) -> !fir.array<?xf32>

// CHECK-LABEL:  func private @derivedfunc(!fir.ref<!fir.type<t{x:f32}>>, f32)
// CHECK-BOX-LABEL:  func private @derivedfunc(!fir.box<!fir.type<t{x:f32}>>, f32)
func.func private @derivedfunc(f32) -> !fir.type<t{x:f32}>

// CHECK-LABEL:  func private @boxfunc(!fir.ref<!fir.box<!fir.heap<f64>>>, i64)
// CHECK-BOX-LABEL:  func private @boxfunc(!fir.ref<!fir.box<!fir.heap<f64>>>, i64)
func.func private @boxfunc(i64) -> !fir.box<!fir.heap<f64>>


// ------------------------ Test callee rewrite --------------------------------

// CHECK-LABEL:  func private @arrayfunc_callee(
// CHECK-SAME: %[[buffer:.*]]: !fir.ref<!fir.array<?xf32>>, %[[n:.*]]: index) {
// CHECK-BOX-LABEL:  func private @arrayfunc_callee(
// CHECK-BOX-SAME: %[[box:.*]]: !fir.box<!fir.array<?xf32>>, %[[n:.*]]: index) {
func.func private @arrayfunc_callee(%n : index) -> !fir.array<?xf32> {
  %buffer = fir.alloca !fir.array<?xf32>, %n
  // Do something with result (res(4) = 42.)
  %c4 = arith.constant 4 : i64
  %coor = fir.coordinate_of %buffer, %c4 : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  %cst = arith.constant 4.200000e+01 : f32
  fir.store %cst to %coor : !fir.ref<f32>
  %res = fir.load %buffer : !fir.ref<!fir.array<?xf32>>
  return %res : !fir.array<?xf32>

  // CHECK-DAG: %[[coor:.*]] = fir.coordinate_of %[[buffer]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  // CHECK-DAG: fir.store %{{.*}} to %[[coor]] : !fir.ref<f32>
  // CHECK: return

  // CHECK-BOX: %[[buffer:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  // CHECK-BOX-DAG: %[[coor:.*]] = fir.coordinate_of %[[buffer]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  // CHECK-BOX-DAG: fir.store %{{.*}} to %[[coor]] : !fir.ref<f32>
  // CHECK-BOX: return
}


// CHECK-LABEL: func @derivedfunc_callee(
// CHECK-SAME: %[[buffer:.*]]: !fir.ref<!fir.type<t{x:f32}>>, %[[v:.*]]: f32) {
// CHECK-BOX-LABEL: func @derivedfunc_callee(
// CHECK-BOX-SAME: %[[box:.*]]: !fir.box<!fir.type<t{x:f32}>>, %[[v:.*]]: f32) {
func.func @derivedfunc_callee(%v: f32) -> !fir.type<t{x:f32}> {
  %buffer = fir.alloca !fir.type<t{x:f32}>
  %0 = fir.field_index x, !fir.type<t{x:f32}>
  %1 = fir.coordinate_of %buffer, %0 : (!fir.ref<!fir.type<t{x:f32}>>, !fir.field) -> !fir.ref<f32>
  fir.store %v to %1 : !fir.ref<f32>
  %res = fir.load %buffer : !fir.ref<!fir.type<t{x:f32}>>
  return %res : !fir.type<t{x:f32}>

  // CHECK: %[[coor:.*]] = fir.coordinate_of %[[buffer]], %{{.*}} : (!fir.ref<!fir.type<t{x:f32}>>, !fir.field) -> !fir.ref<f32>
  // CHECK: fir.store %[[v]] to %[[coor]] : !fir.ref<f32>
  // CHECK: return

  // CHECK-BOX: %[[buffer:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.type<t{x:f32}>>) -> !fir.ref<!fir.type<t{x:f32}>>
  // CHECK-BOX: %[[coor:.*]] = fir.coordinate_of %[[buffer]], %{{.*}} : (!fir.ref<!fir.type<t{x:f32}>>, !fir.field) -> !fir.ref<f32>
  // CHECK-BOX: fir.store %[[v]] to %[[coor]] : !fir.ref<f32>
  // CHECK-BOX: return
}

// CHECK-LABEL: func @boxfunc_callee(
// CHECK-SAME: %[[buffer:.*]]: !fir.ref<!fir.box<!fir.heap<f64>>>) {
// CHECK-BOX-LABEL: func @boxfunc_callee(
// CHECK-BOX-SAME: %[[buffer:.*]]: !fir.ref<!fir.box<!fir.heap<f64>>>) {
func.func @boxfunc_callee() -> !fir.box<!fir.heap<f64>> {
  %alloc = fir.allocmem f64
  %res = fir.embox %alloc : (!fir.heap<f64>) -> !fir.box<!fir.heap<f64>>
  return %res : !fir.box<!fir.heap<f64>>
  // CHECK: %[[box:.*]] = fir.embox %{{.*}} : (!fir.heap<f64>) -> !fir.box<!fir.heap<f64>>
  // CHECK: fir.store %[[box]] to %[[buffer]] : !fir.ref<!fir.box<!fir.heap<f64>>>
  // CHECK: return

  // CHECK-BOX: %[[box:.*]] = fir.embox %{{.*}} : (!fir.heap<f64>) -> !fir.box<!fir.heap<f64>>
  // CHECK-BOX: fir.store %[[box]] to %[[buffer]] : !fir.ref<!fir.box<!fir.heap<f64>>>
  // CHECK-BOX: return
}

// ------------------------ Test caller rewrite --------------------------------

// CHECK-LABEL: func @call_arrayfunc() {
// CHECK-BOX-LABEL: func @call_arrayfunc() {
func.func @call_arrayfunc() {
  %c100 = arith.constant 100 : index
  %buffer = fir.alloca !fir.array<?xf32>, %c100
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res = fir.call @arrayfunc_callee(%c100) : (index) -> !fir.array<?xf32>
  fir.save_result %res to %buffer(%shape) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
  return

  // CHECK: %[[c100:.*]] = arith.constant 100 : index
  // CHECK: %[[buffer:.*]] = fir.alloca !fir.array<?xf32>, %[[c100]]
  // CHECK: fir.call @arrayfunc_callee(%[[buffer]], %[[c100]]) : (!fir.ref<!fir.array<?xf32>>, index) -> ()
  // CHECK-NOT: fir.save_result

  // CHECK-BOX: %[[c100:.*]] = arith.constant 100 : index
  // CHECK-BOX: %[[buffer:.*]] = fir.alloca !fir.array<?xf32>, %[[c100]]
  // CHECK-BOX: %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
  // CHECK-BOX: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  // CHECK-BOX: fir.call @arrayfunc_callee(%[[box]], %[[c100]]) : (!fir.box<!fir.array<?xf32>>, index) -> ()
  // CHECK-BOX-NOT: fir.save_result
}

// CHECK-LABEL: func @call_derivedfunc() {
// CHECK-BOX-LABEL: func @call_derivedfunc() {
func.func @call_derivedfunc() {
  %buffer = fir.alloca !fir.type<t{x:f32}>
  %cst = arith.constant 4.200000e+01 : f32
  %res = fir.call @derivedfunc_callee(%cst) : (f32) -> !fir.type<t{x:f32}>
  fir.save_result %res to %buffer : !fir.type<t{x:f32}>, !fir.ref<!fir.type<t{x:f32}>>
  return
  // CHECK: %[[buffer:.*]] = fir.alloca !fir.type<t{x:f32}>
  // CHECK: %[[cst:.*]] = arith.constant {{.*}} : f32
  // CHECK: fir.call @derivedfunc_callee(%[[buffer]], %[[cst]]) : (!fir.ref<!fir.type<t{x:f32}>>, f32) -> ()
  // CHECK-NOT: fir.save_result

  // CHECK-BOX: %[[buffer:.*]] = fir.alloca !fir.type<t{x:f32}>
  // CHECK-BOX: %[[cst:.*]] = arith.constant {{.*}} : f32
  // CHECK-BOX: %[[box:.*]] = fir.embox %[[buffer]] : (!fir.ref<!fir.type<t{x:f32}>>) -> !fir.box<!fir.type<t{x:f32}>>
  // CHECK-BOX: fir.call @derivedfunc_callee(%[[box]], %[[cst]]) : (!fir.box<!fir.type<t{x:f32}>>, f32) -> ()
  // CHECK-BOX-NOT: fir.save_result
}

func.func private @derived_lparams_func() -> !fir.type<t2(l1:i32,l2:i32){x:f32}>

// CHECK-LABEL: func @call_derived_lparams_func(
// CHECK-SAME: %[[buffer:.*]]: !fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>
// CHECK-BOX-LABEL: func @call_derived_lparams_func(
// CHECK-BOX-SAME: %[[buffer:.*]]: !fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>
func.func @call_derived_lparams_func(%buffer: !fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>) {
  %l1 = arith.constant 3 : i32
  %l2 = arith.constant 5 : i32
  %res = fir.call @derived_lparams_func() : () -> !fir.type<t2(l1:i32,l2:i32){x:f32}>
  fir.save_result %res to %buffer typeparams %l1, %l2 : !fir.type<t2(l1:i32,l2:i32){x:f32}>, !fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>, i32, i32
  return

  // CHECK: %[[l1:.*]] = arith.constant 3 : i32
  // CHECK: %[[l2:.*]] = arith.constant 5 : i32
  // CHECK: fir.call @derived_lparams_func(%[[buffer]]) : (!fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>) -> ()
  // CHECK-NOT: fir.save_result

  // CHECK-BOX: %[[l1:.*]] = arith.constant 3 : i32
  // CHECK-BOX: %[[l2:.*]] = arith.constant 5 : i32
  // CHECK-BOX: %[[box:.*]] = fir.embox %[[buffer]] typeparams %[[l1]], %[[l2]] : (!fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>, i32, i32) -> !fir.box<!fir.type<t2(l1:i32,l2:i32){x:f32}>>
  // CHECK-BOX: fir.call @derived_lparams_func(%[[box]]) : (!fir.box<!fir.type<t2(l1:i32,l2:i32){x:f32}>>) -> ()
  // CHECK-BOX-NOT: fir.save_result
}

// CHECK-LABEL: func @call_boxfunc() {
// CHECK-BOX-LABEL: func @call_boxfunc() {
func.func @call_boxfunc() {
  %buffer = fir.alloca !fir.box<!fir.heap<f64>>
  %res = fir.call @boxfunc_callee() : () -> !fir.box<!fir.heap<f64>>
  fir.save_result %res to %buffer: !fir.box<!fir.heap<f64>>, !fir.ref<!fir.box<!fir.heap<f64>>>
  return

  // CHECK: %[[buffer:.*]] = fir.alloca !fir.box<!fir.heap<f64>>
  // CHECK: fir.call @boxfunc_callee(%[[buffer]]) : (!fir.ref<!fir.box<!fir.heap<f64>>>) -> ()
  // CHECK-NOT: fir.save_result

  // CHECK-BOX: %[[buffer:.*]] = fir.alloca !fir.box<!fir.heap<f64>>
  // CHECK-BOX: fir.call @boxfunc_callee(%[[buffer]]) : (!fir.ref<!fir.box<!fir.heap<f64>>>) -> ()
  // CHECK-BOX-NOT: fir.save_result
}

func.func private @chararrayfunc(index, index) -> !fir.array<?x!fir.char<1,?>>

// CHECK-LABEL: func @call_chararrayfunc() {
// CHECK-BOX-LABEL: func @call_chararrayfunc() {
func.func @call_chararrayfunc() {
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %buffer = fir.alloca !fir.array<?x!fir.char<1,?>>(%c100 : index), %c50
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res = fir.call @chararrayfunc(%c100, %c50) : (index, index) -> !fir.array<?x!fir.char<1,?>>
  fir.save_result %res to %buffer(%shape) typeparams %c50 : !fir.array<?x!fir.char<1,?>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index
  return

  // CHECK: %[[c100:.*]] = arith.constant 100 : index
  // CHECK: %[[c50:.*]] = arith.constant 50 : index
  // CHECK: %[[buffer:.*]] = fir.alloca !fir.array<?x!fir.char<1,?>>(%[[c100]] : index), %[[c50]]
  // CHECK: fir.call @chararrayfunc(%[[buffer]], %[[c100]], %[[c50]]) : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, index, index) -> ()
  // CHECK-NOT: fir.save_result

  // CHECK-BOX: %[[c100:.*]] = arith.constant 100 : index
  // CHECK-BOX: %[[c50:.*]] = arith.constant 50 : index
  // CHECK-BOX: %[[buffer:.*]] = fir.alloca !fir.array<?x!fir.char<1,?>>(%[[c100]] : index), %[[c50]]
  // CHECK-BOX: %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
  // CHECK-BOX: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) typeparams %[[c50]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  // CHECK-BOX: fir.call @chararrayfunc(%[[box]], %[[c100]], %[[c50]]) : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> ()
  // CHECK-BOX-NOT: fir.save_result
}

// ------------------------ Test fir.address_of rewrite ------------------------

func.func private @takesfuncarray((i32) -> !fir.array<?xf32>)

// CHECK-LABEL: func @test_address_of() {
// CHECK-BOX-LABEL: func @test_address_of() {
func.func @test_address_of() {
  %0 = fir.address_of(@arrayfunc) : (i32) -> !fir.array<?xf32>
  fir.call @takesfuncarray(%0) : ((i32) -> !fir.array<?xf32>) -> ()
  return

  // CHECK: %[[addrOf:.*]] = fir.address_of(@arrayfunc) : (!fir.ref<!fir.array<?xf32>>, i32) -> ()
  // CHECK: %[[conv:.*]] = fir.convert %[[addrOf]] : ((!fir.ref<!fir.array<?xf32>>, i32) -> ()) -> ((i32) -> !fir.array<?xf32>)
  // CHECK: fir.call @takesfuncarray(%[[conv]]) : ((i32) -> !fir.array<?xf32>) -> ()

  // CHECK-BOX: %[[addrOf:.*]] = fir.address_of(@arrayfunc) : (!fir.box<!fir.array<?xf32>>, i32) -> ()
  // CHECK-BOX: %[[conv:.*]] = fir.convert %[[addrOf]] : ((!fir.box<!fir.array<?xf32>>, i32) -> ()) -> ((i32) -> !fir.array<?xf32>)
  // CHECK-BOX: fir.call @takesfuncarray(%[[conv]]) : ((i32) -> !fir.array<?xf32>) -> ()

}

// ----------------------- Test indirect calls rewrite ------------------------

// CHECK-LABEL: func @test_indirect_calls(
// CHECK-SAME: %[[arg0:.*]]: () -> ()) {
// CHECK-BOX-LABEL: func @test_indirect_calls(
// CHECK-BOX-SAME: %[[arg0:.*]]: () -> ()) {
func.func @test_indirect_calls(%arg0: () -> ()) {
  %c100 = arith.constant 100 : index
  %buffer = fir.alloca !fir.array<?xf32>, %c100
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %0 = fir.convert %arg0 : (() -> ()) -> ((index) -> !fir.array<?xf32>)
  %res = fir.call %0(%c100) : (index) -> !fir.array<?xf32>
  fir.save_result %res to %buffer(%shape) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
  return

  // CHECK: %[[c100:.*]] = arith.constant 100 : index
  // CHECK: %[[buffer:.*]] = fir.alloca !fir.array<?xf32>, %[[c100]]
  // CHECK: %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
  // CHECK: %[[original_conv:.*]] = fir.convert %[[arg0]] : (() -> ()) -> ((index) -> !fir.array<?xf32>)
  // CHECK: %[[conv:.*]] = fir.convert %[[original_conv]] : ((index) -> !fir.array<?xf32>) -> ((!fir.ref<!fir.array<?xf32>>, index) -> ())
  // CHECK: fir.call %[[conv]](%[[buffer]], %c100) : (!fir.ref<!fir.array<?xf32>>, index) -> ()
  // CHECK-NOT: fir.save_result

  // CHECK-BOX: %[[c100:.*]] = arith.constant 100 : index
  // CHECK-BOX: %[[buffer:.*]] = fir.alloca !fir.array<?xf32>, %[[c100]]
  // CHECK-BOX: %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
  // CHECK-BOX: %[[original_conv:.*]] = fir.convert %[[arg0]] : (() -> ()) -> ((index) -> !fir.array<?xf32>)
  // CHECK-BOX: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  // CHECK-BOX: %[[conv:.*]] = fir.convert %[[original_conv]] : ((index) -> !fir.array<?xf32>) -> ((!fir.box<!fir.array<?xf32>>, index) -> ())
  // CHECK-BOX: fir.call %[[conv]](%[[box]], %c100) : (!fir.box<!fir.array<?xf32>>, index) -> ()
  // CHECK-BOX-NOT: fir.save_result
}
