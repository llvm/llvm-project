! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Test fir.dummy_scope assignment to argument x
subroutine sub_arg(x)
  integer :: x
end subroutine sub_arg
! CHECK-LABEL:   func.func @_QPsub_arg(
! CHECK-SAME:                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFsub_argEx"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           return
! CHECK:         }

! Test fir.dummy_scope is created even when there are no arguments.
subroutine sub_noarg
end subroutine sub_noarg
! CHECK-LABEL:   func.func @_QPsub_noarg() {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           return
! CHECK:         }

! Test fir.dummy_scope assignment to argument x
function func_arg(x)
  integer :: x, func_arg
  func_arg = x
end function func_arg
! CHECK-LABEL:   func.func @_QPfunc_arg(
! CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "x"}) -> i32 {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "func_arg", uniq_name = "_QFfunc_argEfunc_arg"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFfunc_argEfunc_arg"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFfunc_argEx"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           return %[[VAL_6]] : i32
! CHECK:         }

! Test fir.dummy_scope is created even when there are no arguments.
function func_noarg
  integer :: func_noarg
  func_noarg = 1
end function func_noarg
! CHECK-LABEL:   func.func @_QPfunc_noarg() -> i32 {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "func_noarg", uniq_name = "_QFfunc_noargEfunc_noarg"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFfunc_noargEfunc_noarg"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:           hlfir.assign %[[VAL_3]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           return %[[VAL_4]] : i32
! CHECK:         }
