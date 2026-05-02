! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

module m1
  external :: sub
  external :: fun

  type ty
    procedure(), pointer, nopass :: ptr5 => fun
  end type

  type t
    procedure(), pointer, nopass :: ptr5 => sub
  end type

  procedure(), pointer :: ptr6 => sub
end module

use m1
integer :: jj = 4
call ptr6(10)
call ptr5(10)
print *, "Pass"
end

subroutine sub(a)
  integer :: a
  print *, "sub"
end subroutine

integer function fun(a)
  integer :: a
  print *, "fun"
  fun = a * 2
end function

! CHECK-LABEL: func.func @_QQmain() {
! CHECK:         %[[PTR6_ADDR:.*]] = fir.address_of(@_QMm1Eptr6) : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:         %[[PTR6:.*]]:2 = hlfir.declare %[[PTR6_ADDR]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMm1Eptr6"}
! CHECK:         %[[C10:.*]] = arith.constant 10 : i32
! CHECK:         %[[TMP:.*]]:3 = hlfir.associate %[[C10]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:         %[[P:.*]] = fir.load %[[PTR6]]#0 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:         %[[F:.*]] = fir.box_addr %[[P]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<i32>) -> ())
! CHECK:         fir.call %[[F]](%[[TMP]]#0) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:         hlfir.end_associate %[[TMP]]#1, %[[TMP]]#2 : !fir.ref<i32>, i1

! CHECK-LABEL: func.func @_QPsub(%arg0: !fir.ref<i32> {fir.bindc_name = "a"}) {
! CHECK:         %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[A:.*]]:2 = hlfir.declare %arg0 dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFsubEa"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         return

! CHECK-LABEL: func.func @_QPfun(%arg0: !fir.ref<i32> {fir.bindc_name = "a"}) -> i32 {
! CHECK:         %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[A:.*]]:2 = hlfir.declare %arg0 dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFfunEa"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[FUN_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "fun", uniq_name = "_QFfunEfun"}
! CHECK:         %[[FUN:.*]]:2 = hlfir.declare %[[FUN_ALLOCA]] {uniq_name = "_QFfunEfun"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[C2:.*]] = arith.constant 2 : i32
! CHECK:         %[[LOAD_A:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:         %[[MUL:.*]] = arith.muli %[[C2]], %[[LOAD_A]] : i32
! CHECK:         hlfir.assign %[[MUL]] to %[[FUN]]#0 : i32, !fir.ref<i32>
! CHECK:         %[[RESULT:.*]] = fir.load %[[FUN]]#0 : !fir.ref<i32>
! CHECK:         return %[[RESULT]] : i32

! CHECK-LABEL: fir.global @_QMm1Eptr6 : !fir.boxproc<() -> ()> {
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QPsub) : (!fir.ref<i32>) -> ()
! CHECK:         %[[BOX:.*]] = fir.emboxproc %[[ADDR]] : ((!fir.ref<i32>) -> ()) -> !fir.boxproc<() -> ()>
! CHECK:         fir.has_value %[[BOX]] : !fir.boxproc<() -> ()>
