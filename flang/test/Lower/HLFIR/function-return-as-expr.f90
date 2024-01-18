! RUN: bbc -emit-hlfir --polymorphic-type -o - %s -I nowhere 2>&1 | FileCheck %s

module types
  type t1
  end type t1
end module types

subroutine test1
  integer :: i
  i = inner() + 1
contains
  function inner()
    integer, allocatable ::  inner
  end function inner
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = ".result"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.heap<i32>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32

subroutine test2
  character(len=:), allocatable :: c
  c = inner()
contains
  function inner()
    character(len=:), allocatable ::  inner
  end function inner
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2() {
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_11:.*]] = fir.box_elesize %[[VAL_10]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_12:.*]] = fir.emboxchar %[[VAL_9]], %[[VAL_11]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_13:.*]] = arith.constant false
! CHECK:           %[[VAL_14:.*]] = hlfir.as_expr %[[VAL_12]] move %[[VAL_13]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:           hlfir.assign %[[VAL_14]] to %{{.*}}#0 realloc : !hlfir.expr<!fir.char<1,?>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>

subroutine test3
  character(len=:), allocatable :: c
  c = inner()
contains
  function inner()
    character(len=3), allocatable ::  inner
  end function inner
end subroutine test3
! CHECK-LABEL:   func.func @_QPtest3() {
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>)
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.heap<!fir.char<1,3>>>) -> !fir.heap<!fir.char<1,3>>
! CHECK:           %[[VAL_16:.*]] = arith.constant false
! CHECK:           %[[VAL_17:.*]] = hlfir.as_expr %[[VAL_15]] move %[[VAL_16]] : (!fir.heap<!fir.char<1,3>>, i1) -> !hlfir.expr<!fir.char<1,3>>
! CHECK:           hlfir.assign %[[VAL_17]] to %{{.*}}#0 realloc : !hlfir.expr<!fir.char<1,3>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>

subroutine test4
  class(*), allocatable :: p
  p = inner()
contains
  function inner()
    class(*), allocatable :: inner
  end function inner
end subroutine test4
! CHECK-LABEL:   func.func @_QPtest4() {
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.class<!fir.heap<none>>>) -> (!fir.ref<!fir.class<!fir.heap<none>>>, !fir.ref<!fir.class<!fir.heap<none>>>)
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK:           %[[VAL_8:.*]] = arith.constant false
! CHECK:           %[[VAL_9:.*]] = hlfir.as_expr %[[VAL_7]] move %[[VAL_8]] : (!fir.class<!fir.heap<none>>, i1) -> !hlfir.expr<none?>
! CHECK:           hlfir.assign %[[VAL_9]] to %{{.*}}#0 realloc : !hlfir.expr<none?>, !fir.ref<!fir.class<!fir.heap<none>>>

subroutine test5
  use types
  type(t1) :: r
  r = inner()
contains
  function inner()
    type(t1) :: inner
  end function inner
end subroutine test5
! CHECK-LABEL:   func.func @_QPtest5() {
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTt1>>) -> (!fir.ref<!fir.type<_QMtypesTt1>>, !fir.ref<!fir.type<_QMtypesTt1>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant false
! CHECK:           %[[VAL_6:.*]] = hlfir.as_expr %[[VAL_4]]#0 move %[[VAL_5]] : (!fir.ref<!fir.type<_QMtypesTt1>>, i1) -> !hlfir.expr<!fir.type<_QMtypesTt1>>
! CHECK:           hlfir.assign %[[VAL_6]] to %{{.*}}#0 : !hlfir.expr<!fir.type<_QMtypesTt1>>, !fir.ref<!fir.type<_QMtypesTt1>>

subroutine test6(x)
  character(len=:), allocatable :: c(:)
  integer :: x(:)
  c = inner(x)
contains
  elemental function inner(x)
    integer, intent(in) ::  x
    character(len=3) ::  inner
  end function inner
end subroutine test6
! CHECK-LABEL:   func.func @_QPtest6(
! CHECK:           %[[VAL_14:.*]] = hlfir.elemental
! CHECK:             %[[VAL_24:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
! CHECK:             %[[VAL_25:.*]] = arith.constant false
! CHECK:             %[[VAL_26:.*]] = hlfir.as_expr %[[VAL_24]]#0 move %[[VAL_25]] : (!fir.ref<!fir.char<1,3>>, i1) -> !hlfir.expr<!fir.char<1,3>>
! CHECK:             hlfir.yield_element %[[VAL_26]] : !hlfir.expr<!fir.char<1,3>>

subroutine test7(x)
  use types
  integer :: x(:)
  class(*), allocatable :: p(:)
  p = inner(x)
contains
  elemental function inner(x)
    integer, intent(in) :: x
    type(t1) :: inner
  end function inner
end subroutine test7
! CHECK-LABEL:   func.func @_QPtest7(
! CHECK:           %[[VAL_12:.*]] = hlfir.elemental
! CHECK:             %[[VAL_16:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTt1>>) -> (!fir.ref<!fir.type<_QMtypesTt1>>, !fir.ref<!fir.type<_QMtypesTt1>>)
! CHECK:             %[[VAL_17:.*]] = arith.constant false
! CHECK:             %[[VAL_18:.*]] = hlfir.as_expr %[[VAL_16]]#0 move %[[VAL_17]] : (!fir.ref<!fir.type<_QMtypesTt1>>, i1) -> !hlfir.expr<!fir.type<_QMtypesTt1>>
! CHECK:             hlfir.yield_element %[[VAL_18]] : !hlfir.expr<!fir.type<_QMtypesTt1>>

subroutine test8
  if (associated(inner())) STOP 1
contains
  function inner()
    real, pointer :: inner
  end function inner
end subroutine test8
! CHECK-LABEL:   func.func @_QPtest8() {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
