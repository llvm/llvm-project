// Test rewrite of functions that return fir.array<>, fir.type<>, fir.box<> to
// functions that take an additional argument for the result.

// RUN: fir-opt %s --abstract-result-on-func-opt | FileCheck %s --check-prefix=FUNC-REF
// RUN: fir-opt %s --abstract-result-on-func-opt=abstract-result-as-box | FileCheck %s --check-prefix=FUNC-BOX
// RUN: fir-opt %s --abstract-result-on-global-opt | FileCheck %s --check-prefix=GLOBAL-REF
// RUN: fir-opt %s --abstract-result-on-global-opt=abstract-result-as-box | FileCheck %s --check-prefix=GLOBAL-BOX

// ----------------------- Test declaration rewrite ----------------------------

// FUNC-REF-LABEL:  func private @arrayfunc(!fir.ref<!fir.array<?xf32>>, i32)
// FUNC-BOX-LABEL:  func private @arrayfunc(!fir.box<!fir.array<?xf32>>, i32)
func.func private @arrayfunc(i32) -> !fir.array<?xf32>

// FUNC-REF-LABEL:  func private @derivedfunc(!fir.ref<!fir.type<t{x:f32}>>, f32)
// FUNC-BOX-LABEL:  func private @derivedfunc(!fir.box<!fir.type<t{x:f32}>>, f32)
func.func private @derivedfunc(f32) -> !fir.type<t{x:f32}>

// FUNC-REF-LABEL:  func private @boxfunc(!fir.ref<!fir.box<!fir.heap<f64>>>, i64)
// FUNC-BOX-LABEL:  func private @boxfunc(!fir.ref<!fir.box<!fir.heap<f64>>>, i64)
func.func private @boxfunc(i64) -> !fir.box<!fir.heap<f64>>


// ------------------------ Test callee rewrite --------------------------------

// FUNC-REF-LABEL:  func private @arrayfunc_callee(
// FUNC-REF-SAME: %[[buffer:.*]]: !fir.ref<!fir.array<?xf32>>, %[[n:.*]]: index) {
// FUNC-BOX-LABEL:  func private @arrayfunc_callee(
// FUNC-BOX-SAME: %[[box:.*]]: !fir.box<!fir.array<?xf32>>, %[[n:.*]]: index) {
func.func private @arrayfunc_callee(%n : index) -> !fir.array<?xf32> {
  %buffer = fir.alloca !fir.array<?xf32>, %n
  // Do something with result (res(4) = 42.)
  %c4 = arith.constant 4 : i64
  %coor = fir.coordinate_of %buffer, %c4 : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  %cst = arith.constant 4.200000e+01 : f32
  fir.store %cst to %coor : !fir.ref<f32>
  %res = fir.load %buffer : !fir.ref<!fir.array<?xf32>>
  return %res : !fir.array<?xf32>

  // FUNC-REF-DAG: %[[coor:.*]] = fir.coordinate_of %[[buffer]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  // FUNC-REF-DAG: fir.store %{{.*}} to %[[coor]] : !fir.ref<f32>
  // FUNC-REF: return

  // FUNC-BOX: %[[buffer:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  // FUNC-BOX-DAG: %[[coor:.*]] = fir.coordinate_of %[[buffer]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  // FUNC-BOX-DAG: fir.store %{{.*}} to %[[coor]] : !fir.ref<f32>
  // FUNC-BOX: return
}


// FUNC-REF-LABEL: func @derivedfunc_callee(
// FUNC-REF-SAME: %[[buffer:.*]]: !fir.ref<!fir.type<t{x:f32}>>, %[[v:.*]]: f32) {
// FUNC-BOX-LABEL: func @derivedfunc_callee(
// FUNC-BOX-SAME: %[[box:.*]]: !fir.box<!fir.type<t{x:f32}>>, %[[v:.*]]: f32) {
func.func @derivedfunc_callee(%v: f32) -> !fir.type<t{x:f32}> {
  %buffer = fir.alloca !fir.type<t{x:f32}>
  %0 = fir.field_index x, !fir.type<t{x:f32}>
  %1 = fir.coordinate_of %buffer, %0 : (!fir.ref<!fir.type<t{x:f32}>>, !fir.field) -> !fir.ref<f32>
  fir.store %v to %1 : !fir.ref<f32>
  %res = fir.load %buffer : !fir.ref<!fir.type<t{x:f32}>>
  return %res : !fir.type<t{x:f32}>

  // FUNC-REF: %[[coor:.*]] = fir.coordinate_of %[[buffer]], %{{.*}} : (!fir.ref<!fir.type<t{x:f32}>>, !fir.field) -> !fir.ref<f32>
  // FUNC-REF: fir.store %[[v]] to %[[coor]] : !fir.ref<f32>
  // FUNC-REF: return

  // FUNC-BOX: %[[buffer:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.type<t{x:f32}>>) -> !fir.ref<!fir.type<t{x:f32}>>
  // FUNC-BOX: %[[coor:.*]] = fir.coordinate_of %[[buffer]], %{{.*}} : (!fir.ref<!fir.type<t{x:f32}>>, !fir.field) -> !fir.ref<f32>
  // FUNC-BOX: fir.store %[[v]] to %[[coor]] : !fir.ref<f32>
  // FUNC-BOX: return
}

// FUNC-REF-LABEL: func @boxfunc_callee(
// FUNC-REF-SAME: %[[buffer:.*]]: !fir.ref<!fir.box<!fir.heap<f64>>>) {
// FUNC-BOX-LABEL: func @boxfunc_callee(
// FUNC-BOX-SAME: %[[buffer:.*]]: !fir.ref<!fir.box<!fir.heap<f64>>>) {
func.func @boxfunc_callee() -> !fir.box<!fir.heap<f64>> {
  %alloc = fir.allocmem f64
  %res = fir.embox %alloc : (!fir.heap<f64>) -> !fir.box<!fir.heap<f64>>
  return %res : !fir.box<!fir.heap<f64>>
  // FUNC-REF: %[[box:.*]] = fir.embox %{{.*}} : (!fir.heap<f64>) -> !fir.box<!fir.heap<f64>>
  // FUNC-REF: fir.store %[[box]] to %[[buffer]] : !fir.ref<!fir.box<!fir.heap<f64>>>
  // FUNC-REF: return

  // FUNC-BOX: %[[box:.*]] = fir.embox %{{.*}} : (!fir.heap<f64>) -> !fir.box<!fir.heap<f64>>
  // FUNC-BOX: fir.store %[[box]] to %[[buffer]] : !fir.ref<!fir.box<!fir.heap<f64>>>
  // FUNC-BOX: return
}

// ------------------------ Test caller rewrite --------------------------------

// FUNC-REF-LABEL: func @call_arrayfunc() {
// FUNC-BOX-LABEL: func @call_arrayfunc() {
func.func @call_arrayfunc() {
  %c100 = arith.constant 100 : index
  %buffer = fir.alloca !fir.array<?xf32>, %c100
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res = fir.call @arrayfunc_callee(%c100) : (index) -> !fir.array<?xf32>
  fir.save_result %res to %buffer(%shape) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
  return

  // FUNC-REF: %[[c100:.*]] = arith.constant 100 : index
  // FUNC-REF: %[[buffer:.*]] = fir.alloca !fir.array<?xf32>, %[[c100]]
  // FUNC-REF: fir.call @arrayfunc_callee(%[[buffer]], %[[c100]]) : (!fir.ref<!fir.array<?xf32>>, index) -> ()
  // FUNC-REF-NOT: fir.save_result

  // FUNC-BOX: %[[c100:.*]] = arith.constant 100 : index
  // FUNC-BOX: %[[buffer:.*]] = fir.alloca !fir.array<?xf32>, %[[c100]]
  // FUNC-BOX: %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
  // FUNC-BOX: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  // FUNC-BOX: fir.call @arrayfunc_callee(%[[box]], %[[c100]]) : (!fir.box<!fir.array<?xf32>>, index) -> ()
  // FUNC-BOX-NOT: fir.save_result
}

// FUNC-REF-LABEL: func @call_derivedfunc() {
// FUNC-BOX-LABEL: func @call_derivedfunc() {
func.func @call_derivedfunc() {
  %buffer = fir.alloca !fir.type<t{x:f32}>
  %cst = arith.constant 4.200000e+01 : f32
  %res = fir.call @derivedfunc_callee(%cst) : (f32) -> !fir.type<t{x:f32}>
  fir.save_result %res to %buffer : !fir.type<t{x:f32}>, !fir.ref<!fir.type<t{x:f32}>>
  return
  // FUNC-REF: %[[buffer:.*]] = fir.alloca !fir.type<t{x:f32}>
  // FUNC-REF: %[[cst:.*]] = arith.constant {{.*}} : f32
  // FUNC-REF: fir.call @derivedfunc_callee(%[[buffer]], %[[cst]]) : (!fir.ref<!fir.type<t{x:f32}>>, f32) -> ()
  // FUNC-REF-NOT: fir.save_result

  // FUNC-BOX: %[[buffer:.*]] = fir.alloca !fir.type<t{x:f32}>
  // FUNC-BOX: %[[cst:.*]] = arith.constant {{.*}} : f32
  // FUNC-BOX: %[[box:.*]] = fir.embox %[[buffer]] : (!fir.ref<!fir.type<t{x:f32}>>) -> !fir.box<!fir.type<t{x:f32}>>
  // FUNC-BOX: fir.call @derivedfunc_callee(%[[box]], %[[cst]]) : (!fir.box<!fir.type<t{x:f32}>>, f32) -> ()
  // FUNC-BOX-NOT: fir.save_result
}

func.func private @derived_lparams_func() -> !fir.type<t2(l1:i32,l2:i32){x:f32}>

// FUNC-REF-LABEL: func @call_derived_lparams_func(
// FUNC-REF-SAME: %[[buffer:.*]]: !fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>
// FUNC-BOX-LABEL: func @call_derived_lparams_func(
// FUNC-BOX-SAME: %[[buffer:.*]]: !fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>
func.func @call_derived_lparams_func(%buffer: !fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>) {
  %l1 = arith.constant 3 : i32
  %l2 = arith.constant 5 : i32
  %res = fir.call @derived_lparams_func() : () -> !fir.type<t2(l1:i32,l2:i32){x:f32}>
  fir.save_result %res to %buffer typeparams %l1, %l2 : !fir.type<t2(l1:i32,l2:i32){x:f32}>, !fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>, i32, i32
  return

  // FUNC-REF: %[[l1:.*]] = arith.constant 3 : i32
  // FUNC-REF: %[[l2:.*]] = arith.constant 5 : i32
  // FUNC-REF: fir.call @derived_lparams_func(%[[buffer]]) : (!fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>) -> ()
  // FUNC-REF-NOT: fir.save_result

  // FUNC-BOX: %[[l1:.*]] = arith.constant 3 : i32
  // FUNC-BOX: %[[l2:.*]] = arith.constant 5 : i32
  // FUNC-BOX: %[[box:.*]] = fir.embox %[[buffer]] typeparams %[[l1]], %[[l2]] : (!fir.ref<!fir.type<t2(l1:i32,l2:i32){x:f32}>>, i32, i32) -> !fir.box<!fir.type<t2(l1:i32,l2:i32){x:f32}>>
  // FUNC-BOX: fir.call @derived_lparams_func(%[[box]]) : (!fir.box<!fir.type<t2(l1:i32,l2:i32){x:f32}>>) -> ()
  // FUNC-BOX-NOT: fir.save_result
}

// FUNC-REF-LABEL: func @call_boxfunc() {
// FUNC-BOX-LABEL: func @call_boxfunc() {
func.func @call_boxfunc() {
  %buffer = fir.alloca !fir.box<!fir.heap<f64>>
  %res = fir.call @boxfunc_callee() : () -> !fir.box<!fir.heap<f64>>
  fir.save_result %res to %buffer: !fir.box<!fir.heap<f64>>, !fir.ref<!fir.box<!fir.heap<f64>>>
  return

  // FUNC-REF: %[[buffer:.*]] = fir.alloca !fir.box<!fir.heap<f64>>
  // FUNC-REF: fir.call @boxfunc_callee(%[[buffer]]) : (!fir.ref<!fir.box<!fir.heap<f64>>>) -> ()
  // FUNC-REF-NOT: fir.save_result

  // FUNC-BOX: %[[buffer:.*]] = fir.alloca !fir.box<!fir.heap<f64>>
  // FUNC-BOX: fir.call @boxfunc_callee(%[[buffer]]) : (!fir.ref<!fir.box<!fir.heap<f64>>>) -> ()
  // FUNC-BOX-NOT: fir.save_result
}

func.func private @chararrayfunc(index, index) -> !fir.array<?x!fir.char<1,?>>

// FUNC-REF-LABEL: func @call_chararrayfunc() {
// FUNC-BOX-LABEL: func @call_chararrayfunc() {
func.func @call_chararrayfunc() {
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %buffer = fir.alloca !fir.array<?x!fir.char<1,?>>(%c100 : index), %c50
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res = fir.call @chararrayfunc(%c100, %c50) : (index, index) -> !fir.array<?x!fir.char<1,?>>
  fir.save_result %res to %buffer(%shape) typeparams %c50 : !fir.array<?x!fir.char<1,?>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index
  return

  // FUNC-REF: %[[c100:.*]] = arith.constant 100 : index
  // FUNC-REF: %[[c50:.*]] = arith.constant 50 : index
  // FUNC-REF: %[[buffer:.*]] = fir.alloca !fir.array<?x!fir.char<1,?>>(%[[c100]] : index), %[[c50]]
  // FUNC-REF: fir.call @chararrayfunc(%[[buffer]], %[[c100]], %[[c50]]) : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, index, index) -> ()
  // FUNC-REF-NOT: fir.save_result

  // FUNC-BOX: %[[c100:.*]] = arith.constant 100 : index
  // FUNC-BOX: %[[c50:.*]] = arith.constant 50 : index
  // FUNC-BOX: %[[buffer:.*]] = fir.alloca !fir.array<?x!fir.char<1,?>>(%[[c100]] : index), %[[c50]]
  // FUNC-BOX: %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
  // FUNC-BOX: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) typeparams %[[c50]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  // FUNC-BOX: fir.call @chararrayfunc(%[[box]], %[[c100]], %[[c50]]) : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> ()
  // FUNC-BOX-NOT: fir.save_result
}

func.func private @rettcptr() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> attributes {fir.bindc_name = "rettcptr"}

// FUNC-REF-LABEL: func @_QPtest_return_cptr() {
// FUNC-BOX-LABEL: func @_QPtest_return_cptr() {
func.func @_QPtest_return_cptr() {
  // FUNC-REF: [[VAL:.*]] = fir.call @rettcptr() : () -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  // FUNC-BOX: [[VAL:.*]] = fir.call @rettcptr() : () -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  %1 = fir.call @rettcptr() : () -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  return
}

// ------------------------ Test fir.address_of rewrite ------------------------

func.func private @takesfuncarray((i32) -> !fir.array<?xf32>)

// FUNC-REF-LABEL: func @test_address_of() {
// FUNC-BOX-LABEL: func @test_address_of() {
func.func @test_address_of() {
  %0 = fir.address_of(@arrayfunc) : (i32) -> !fir.array<?xf32>
  fir.call @takesfuncarray(%0) : ((i32) -> !fir.array<?xf32>) -> ()
  return

  // FUNC-REF: %[[addrOf:.*]] = fir.address_of(@arrayfunc) : (!fir.ref<!fir.array<?xf32>>, i32) -> ()
  // FUNC-REF: %[[conv:.*]] = fir.convert %[[addrOf]] : ((!fir.ref<!fir.array<?xf32>>, i32) -> ()) -> ((i32) -> !fir.array<?xf32>)
  // FUNC-REF: fir.call @takesfuncarray(%[[conv]]) : ((i32) -> !fir.array<?xf32>) -> ()

  // FUNC-BOX: %[[addrOf:.*]] = fir.address_of(@arrayfunc) : (!fir.box<!fir.array<?xf32>>, i32) -> ()
  // FUNC-BOX: %[[conv:.*]] = fir.convert %[[addrOf]] : ((!fir.box<!fir.array<?xf32>>, i32) -> ()) -> ((i32) -> !fir.array<?xf32>)
  // FUNC-BOX: fir.call @takesfuncarray(%[[conv]]) : ((i32) -> !fir.array<?xf32>) -> ()

}

// ----------------------- Test indirect calls rewrite ------------------------

// FUNC-REF-LABEL: func @test_indirect_calls(
// FUNC-REF-SAME: %[[arg0:.*]]: () -> ()) {
// FUNC-BOX-LABEL: func @test_indirect_calls(
// FUNC-BOX-SAME: %[[arg0:.*]]: () -> ()) {
func.func @test_indirect_calls(%arg0: () -> ()) {
  %c100 = arith.constant 100 : index
  %buffer = fir.alloca !fir.array<?xf32>, %c100
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %0 = fir.convert %arg0 : (() -> ()) -> ((index) -> !fir.array<?xf32>)
  %res = fir.call %0(%c100) : (index) -> !fir.array<?xf32>
  fir.save_result %res to %buffer(%shape) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
  return

  // FUNC-REF: %[[c100:.*]] = arith.constant 100 : index
  // FUNC-REF: %[[buffer:.*]] = fir.alloca !fir.array<?xf32>, %[[c100]]
  // FUNC-REF: %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
  // FUNC-REF: %[[original_conv:.*]] = fir.convert %[[arg0]] : (() -> ()) -> ((index) -> !fir.array<?xf32>)
  // FUNC-REF: %[[conv:.*]] = fir.convert %[[original_conv]] : ((index) -> !fir.array<?xf32>) -> ((!fir.ref<!fir.array<?xf32>>, index) -> ())
  // FUNC-REF: fir.call %[[conv]](%[[buffer]], %c100) : (!fir.ref<!fir.array<?xf32>>, index) -> ()
  // FUNC-REF-NOT: fir.save_result

  // FUNC-BOX: %[[c100:.*]] = arith.constant 100 : index
  // FUNC-BOX: %[[buffer:.*]] = fir.alloca !fir.array<?xf32>, %[[c100]]
  // FUNC-BOX: %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
  // FUNC-BOX: %[[original_conv:.*]] = fir.convert %[[arg0]] : (() -> ()) -> ((index) -> !fir.array<?xf32>)
  // FUNC-BOX: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  // FUNC-BOX: %[[conv:.*]] = fir.convert %[[original_conv]] : ((index) -> !fir.array<?xf32>) -> ((!fir.box<!fir.array<?xf32>>, index) -> ())
  // FUNC-BOX: fir.call %[[conv]](%[[box]], %c100) : (!fir.box<!fir.array<?xf32>>, index) -> ()
  // FUNC-BOX-NOT: fir.save_result
}

// ----------------------- Test GlobalOp rewrite ------------------------

// This is needed to separate GlobalOp tests from FuncOp tests for FileCheck
// FUNC-REF-LABEL: fir.global {{.*}} : {{.*}} {
// FUNC-BOX-LABEL: fir.global {{.*}} : {{.*}} {

// GLOBAL-REF-LABEL: fir.global @global_test_address_of : i32 {
// GLOBAL-BOX-LABEL: fir.global @global_test_address_of : i32 {
fir.global @global_test_address_of : i32 {
  %0 = fir.address_of(@arrayfunc) : (i32) -> !fir.array<?xf32>

  // GLOBAL-REF: %[[addrOf:.*]] = fir.address_of(@arrayfunc) : (!fir.ref<!fir.array<?xf32>>, i32) -> ()
  // GLOBAL-REF: %[[conv:.*]] = fir.convert %[[addrOf]] : ((!fir.ref<!fir.array<?xf32>>, i32) -> ()) -> ((i32) -> !fir.array<?xf32>)

  // GLOBAL-BOX: %[[addrOf:.*]] = fir.address_of(@arrayfunc) : (!fir.box<!fir.array<?xf32>>, i32) -> ()
  // GLOBAL-BOX: %[[conv:.*]] = fir.convert %[[addrOf]] : ((!fir.box<!fir.array<?xf32>>, i32) -> ()) -> ((i32) -> !fir.array<?xf32>)

  %c42 = arith.constant 42 : i32
  fir.has_value %c42 : i32
}
