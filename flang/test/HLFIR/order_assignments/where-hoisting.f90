! Test that scalar expressions are not hoisted from WHERE loops
! when they do not appear
! RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" %s | FileCheck %s

subroutine do_not_hoist_div(n, mask, a)
  integer :: a(10), n
  logical :: mask(10)
  where(mask) a=1/n
end subroutine
! CHECK-LABEL:   func.func @_QPdo_not_hoist_div(
! CHECK-NOT:       arith.divsi
! CHECK:           fir.do_loop {{.*}} {
! CHECK:             fir.if {{.*}} {
! CHECK:               arith.divsi
! CHECK:             }
! CHECK:           }

subroutine do_not_hoist_optional(n, mask, a)
  integer :: a(10)
  integer, optional :: n
  logical :: mask(10)
  where(mask) a=n
end subroutine
! CHECK-LABEL:   func.func @_QPdo_not_hoist_optional(
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}"_QFdo_not_hoist_optionalEn"
! CHECK-NOT:       fir.load %[[VAL_9]]
! CHECK:           fir.do_loop {{.*}} {
! CHECK:             fir.if {{.*}} {
! CHECK:               %[[VAL_15:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:             }
! CHECK:           }

subroutine hoist_function(n, mask, a)
  integer :: a(10, 10)
  integer, optional :: n
  logical :: mask(10, 10)
  forall (i=1:10)
  where(mask(i, :)) a(i,:)=ihoist_me(i)
  end forall
end subroutine
! CHECK-LABEL:   func.func @_QPhoist_function(
! CHECK:           fir.do_loop {{.*}} {
! CHECK:             fir.call @_QPihoist_me
! CHECK:             fir.do_loop {{.*}} {
! CHECK:               fir.if %{{.*}} {
! CHECK-NOT:             fir.call @_QPihoist_me
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK-NOT:       fir.call @_QPihoist_me
