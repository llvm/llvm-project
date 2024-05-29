// Test rewrite of functions that return fir.array<>, fir.type<>, fir.box<> to
// functions that take an additional argument for the result.

// RUN: fir-opt %s --abstract-result | FileCheck %s --check-prefix=FUNC-REF
// RUN: fir-opt %s --abstract-result=abstract-result-as-box | FileCheck %s --check-prefix=FUNC-BOX
// RUN: fir-opt %s --abstract-result | FileCheck %s --check-prefix=GLOBAL-REF
// RUN: fir-opt %s --abstract-result=abstract-result-as-box | FileCheck %s --check-prefix=GLOBAL-BOX

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

// FUNC-REF-LABEL: func @retcptr() -> i64
// FUNC-BOX-LABEL: func @retcptr() -> i64
func.func @retcptr() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {
  %0 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "rec", uniq_name = "_QFrecErec"}
  %1 = fir.load %0 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
  return %1 : !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>

  // FUNC-REF: %[[ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "rec", uniq_name = "_QFrecErec"}
  // FUNC-REF: %[[FIELD:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  // FUNC-REF: %[[ADDR:.*]] = fir.coordinate_of %[[ALLOC]], %[[FIELD]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  // FUNC-REF: %[[VAL:.*]] = fir.load %[[ADDR]] : !fir.ref<i64>
  // FUNC-REF: return %[[VAL]] : i64
  // FUNC-BOX: %[[ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "rec", uniq_name = "_QFrecErec"}
  // FUNC-BOX: %[[FIELD:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  // FUNC-BOX: %[[ADDR:.*]] = fir.coordinate_of %[[ALLOC]], %[[FIELD]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  // FUNC-BOX: %[[VAL:.*]] = fir.load %[[ADDR]] : !fir.ref<i64>
  // FUNC-BOX: return %[[VAL]] : i64
}

// FUNC-REF-LABEL:  func private @arrayfunc_callee_declare(
// FUNC-REF-SAME: %[[buffer:.*]]: !fir.ref<!fir.array<?xf32>>, %[[n:.*]]: index) {
// FUNC-BOX-LABEL:  func private @arrayfunc_callee_declare(
// FUNC-BOX-SAME: %[[box:.*]]: !fir.box<!fir.array<?xf32>>, %[[n:.*]]: index) {
func.func private @arrayfunc_callee_declare(%n : index) -> !fir.array<?xf32> {
  %buffer_alloc = fir.alloca !fir.array<?xf32>, %n
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %buffer = fir.declare %buffer_alloc(%shape) {uniq_name = "x"}: (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  // Do something with result (res(4) = 42.)
  %c4 = arith.constant 4 : i64
  %coor = fir.coordinate_of %buffer, %c4 : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  %cst = arith.constant 4.200000e+01 : f32
  fir.store %cst to %coor : !fir.ref<f32>
  %res = fir.load %buffer : !fir.ref<!fir.array<?xf32>>
  return %res : !fir.array<?xf32>

  // FUNC-REF-DAG: %[[buffer_declare:.*]] = fir.declare %[[buffer]](%{{.*}}) {uniq_name = "x"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  // FUNC-REF-DAG: %[[coor:.*]] = fir.coordinate_of %[[buffer_declare]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  // FUNC-REF-DAG: fir.store %{{.*}} to %[[coor]] : !fir.ref<f32>
  // FUNC-REF: return

  // FUNC-BOX: %[[buffer:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  // FUNC-BOX-DAG: %[[buffer_declare:.*]] = fir.declare %[[buffer]](%{{.*}}) {uniq_name = "x"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  // FUNC-BOX-DAG: %[[coor:.*]] = fir.coordinate_of %[[buffer_declare]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  // FUNC-BOX-DAG: fir.store %{{.*}} to %[[coor]] : !fir.ref<f32>
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

// FUNC-REF-LABEL: func @_QPtest_return_cptr
// FUNC-BOX-LABEL: func @_QPtest_return_cptr
func.func @_QPtest_return_cptr() {
  %0 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = ".result"}
  %1 = fir.call @retcptr() : () -> i64
  %2 = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  %3 = fir.coordinate_of %0, %2 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  fir.store %1 to %3 : !fir.ref<i64>
  return

  // FUNC-REF: %[[ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = ".result"}
  // FUNC-REF: %[[VAL:.*]] = fir.call @retcptr() : () -> i64
  // FUNC-REF: %[[FIELD:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  // FUNC-REF: %[[ADDR:.*]] = fir.coordinate_of %[[ALLOC]], %[[FIELD]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  // FUNC-REF: fir.store %[[VAL]] to %[[ADDR]] : !fir.ref<i64>
  // FUNC-BOX: %[[ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = ".result"}
  // FUNC-BOX: %[[VAL:.*]] = fir.call @retcptr() : () -> i64
  // FUNC-BOX: %[[FIELD:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  // FUNC-BOX: %[[ADDR:.*]] = fir.coordinate_of %[[ALLOC]], %[[FIELD]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  // FUNC-BOX: fir.store %[[VAL]] to %[[ADDR]] : !fir.ref<i64>
}

// FUNC-REF-LABEL: func @dispatch(
// FUNC-REF-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}
// FUNC-BOX-LABEL: func @dispatch(
// FUNC-BOX-SAME:    %[[ARG0:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}
func.func @dispatch(%arg0: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "a"}) {
  %buffer = fir.alloca !fir.type<t{x:f32}>
  %res = fir.dispatch "ret_array"(%arg0 : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%arg0 : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.type<t{x:f32}> {pass_arg_pos = 0 : i32}
  fir.save_result %res to %buffer : !fir.type<t{x:f32}>, !fir.ref<!fir.type<t{x:f32}>>
  return
  // FUNC-REF: %[[buffer:.*]] = fir.alloca !fir.type<t{x:f32}>
  // FUNC-REF: fir.dispatch "ret_array"(%[[ARG0]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%[[buffer]], %[[ARG0]] : !fir.ref<!fir.type<t{x:f32}>>, !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32}
  // FUNC-REF-NOT: fir.save_result

  // FUNC-BOX: %[[buffer:.*]] = fir.alloca !fir.type<t{x:f32}>
  // FUNC-BOX: %[[box:.*]] = fir.embox %[[buffer]] : (!fir.ref<!fir.type<t{x:f32}>>) -> !fir.box<!fir.type<t{x:f32}>>
  // FUNC-BOX: fir.dispatch "ret_array"(%[[ARG0]] : !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) (%[[box]], %[[ARG0]] : !fir.box<!fir.type<t{x:f32}>>, !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32}
  // FUNC-BOX-NOT: fir.save_result
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

// FUNC-REF-LABEL: func.func private @returns_null() -> i64
// FUNC-BOX-LABEL: func.func private @returns_null() -> i64
func.func private @returns_null() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>

// FUNC-REF-LABEL: func @test_address_of_cptr
// FUNC-BOX-LABEL: func @test_address_of_cptr
func.func @test_address_of_cptr() {
  %0 = fir.address_of(@returns_null) : () -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  %1 = fir.convert %0 : (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> (() -> ())
  fir.call @_QMtest_c_func_modPsubr(%1) : (() -> ()) -> ()
  return

  // FUNC-REF: %[[VAL_0:.*]] = fir.address_of(@returns_null) : () -> i64
  // FUNC-REF: %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (() -> i64) -> (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>)
  // FUNC-REF: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> (() -> ())
  // FUNC-REF: fir.call @_QMtest_c_func_modPsubr(%[[VAL_2]]) : (() -> ()) -> ()
  // FUNC-BOX: %[[VAL_0:.*]] = fir.address_of(@returns_null) : () -> i64
  // FUNC-BOX: %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (() -> i64) -> (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>)
  // FUNC-BOX: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> (() -> ())
  // FUNC-BOX: fir.call @_QMtest_c_func_modPsubr(%[[VAL_2]]) : (() -> ()) -> ()
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

// FUNC-REF-LABEL: func @test_indirect_calls_return_cptr(
// FUNC-REF-SAME: %[[ARG0:.*]]: () -> ())
// FUNC-BOX-LABEL: func @test_indirect_calls_return_cptr(
// FUNC-BOX-SAME: %[[ARG0:.*]]: () -> ())
func.func @test_indirect_calls_return_cptr(%arg0: () -> ()) {
  %0 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = ".result"}
  %1 = fir.convert %arg0 : (() -> ()) -> (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>)
  %2 = fir.call %1() : () -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  fir.save_result %2 to %0 : !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
  return

  // FUNC-REF: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = ".result"}
  // FUNC-REF: %[[VAL_1:.*]] = fir.convert %[[ARG0]] : (() -> ()) -> (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>)
  // FUNC-REF: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> (() -> i64)
  // FUNC-REF: %[[VAL_3:.*]] = fir.call %[[VAL_2]]() : () -> i64
  // FUNC-REF: %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  // FUNC-REF: %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  // FUNC-REF: fir.store %[[VAL_3]] to %[[VAL_5]] : !fir.ref<i64>
  // FUNC-BOX: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = ".result"}
  // FUNC-BOX: %[[VAL_1:.*]] = fir.convert %[[ARG0]] : (() -> ()) -> (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>)
  // FUNC-BOX: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (() -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> (() -> i64)
  // FUNC-BOX: %[[VAL_3:.*]] = fir.call %[[VAL_2]]() : () -> i64
  // FUNC-BOX: %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  // FUNC-BOX: %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  // FUNC-BOX: fir.store %[[VAL_3]] to %[[VAL_5]] : !fir.ref<i64>
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
