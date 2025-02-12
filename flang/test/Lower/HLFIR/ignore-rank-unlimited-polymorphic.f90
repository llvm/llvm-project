! Test passing mismatching rank arguments to unlimited polymorphic
! dummy with IGNORE_TKR(R).
! RUN: bbc -emit-hlfir -o - -I nowhere %s 2>&1 | FileCheck %s

module m
  interface
     subroutine callee(x)
       class(*) :: x
       !dir$ ignore_tkr (r) x
     end subroutine callee
  end interface
end module m

subroutine test_integer_scalar
  use m
  integer :: x
  call callee(x)
end subroutine test_integer_scalar
! CHECK-LABEL:   func.func @_QPtest_integer_scalar() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_integer_scalarEx"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_integer_scalarEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<i32>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_3]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_real_explicit_shape_array
  use m
  real :: x(10)
  call callee(x)
end subroutine test_real_explicit_shape_array
! CHECK-LABEL:   func.func @_QPtest_real_explicit_shape_array() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFtest_real_explicit_shape_arrayEx"}
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) {uniq_name = "_QFtest_real_explicit_shape_arrayEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#0(%[[VAL_2]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
! CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_4]] : (!fir.box<!fir.array<10xf32>>) -> !fir.class<!fir.array<10xnone>>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.class<!fir.array<10xnone>>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_6]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_logical_assumed_shape_array(x)
  use m
  logical :: x(:)
  call callee(x)
end subroutine test_logical_assumed_shape_array
! CHECK-LABEL:   func.func @_QPtest_logical_assumed_shape_array(
! CHECK-SAME:                                                   %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_logical_assumed_shape_arrayEx"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:           %[[VAL_2:.*]] = fir.rebox %[[VAL_1]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.class<!fir.array<?xnone>>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_3]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_real_2d_pointer(x)
  use m
  real, pointer :: x(:, :)
  call callee(x)
end subroutine test_real_2d_pointer
! CHECK-LABEL:   func.func @_QPtest_real_2d_pointer(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_real_2d_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>) -> !fir.class<!fir.array<?x?xnone>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_4]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_up_assumed_shape_1d_array(x)
  use m
  class(*) :: x(:)
  call callee(x)
end subroutine test_up_assumed_shape_1d_array
! CHECK-LABEL:   func.func @_QPtest_up_assumed_shape_1d_array(
! CHECK-SAME:                                                 %[[VAL_0:.*]]: !fir.class<!fir.array<?xnone>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_up_assumed_shape_1d_arrayEx"} : (!fir.class<!fir.array<?xnone>>, !fir.dscope) -> (!fir.class<!fir.array<?xnone>>, !fir.class<!fir.array<?xnone>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.class<!fir.array<?xnone>>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_2]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_derived_explicit_shape_array
  use m
  type t1
     real, allocatable :: a
  end type t1
  type(t1) :: x(10)
  call callee(x)
end subroutine test_derived_explicit_shape_array
! CHECK-LABEL:   func.func @_QPtest_derived_explicit_shape_array() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>> {bindc_name = "x", uniq_name = "_QFtest_derived_explicit_shape_arrayEx"}
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) {uniq_name = "_QFtest_derived_explicit_shape_arrayEx"} : (!fir.ref<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>, !fir.ref<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>)
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_3]]#1(%[[VAL_4]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>) -> !fir.box<none>
! CHECK:           fir.call @_FortranAInitialize(%[[VAL_8]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
! CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_3]]#0(%[[VAL_2]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<10x!fir.type<_QFtest_derived_explicit_shape_arrayTt1{a:!fir.box<!fir.heap<f32>>}>>>) -> !fir.class<!fir.array<10xnone>>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (!fir.class<!fir.array<10xnone>>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_13]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_up_allocatable_2d_array(x)
  use m
  class(*), allocatable :: x(:, :)
  call callee(x)
end subroutine test_up_allocatable_2d_array
! CHECK-LABEL:   func.func @_QPtest_up_allocatable_2d_array(
! CHECK-SAME:                                               %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_up_allocatable_2d_arrayEx"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.class<!fir.array<?x?xnone>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_4]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_up_pointer_1d_array(x)
  use m
  class(*), pointer :: x(:)
  call callee(x)
end subroutine test_up_pointer_1d_array
! CHECK-LABEL:   func.func @_QPtest_up_pointer_1d_array(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_up_pointer_1d_arrayEx"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.class<!fir.ptr<!fir.array<?xnone>>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.class<!fir.array<?xnone>>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_4]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           return
! CHECK:         }
