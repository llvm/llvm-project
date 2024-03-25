! Test lowering of of expressions as address
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPfoo(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>
subroutine foo(x)
  integer :: x
  read (*,*) x
  ! CHECK: %[[x:.]]:2 = hlfir.declare %[[arg0]] {uniq_name = "_QFfooEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[x_cast:.*]] = fir.convert %[[x]]#1 : (!fir.ref<i32>) -> !fir.ref<i64>
  ! CHECK: fir.call @_FortranAioInputInteger(%{{.*}}, %[[x_cast]], %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
end subroutine

subroutine expr_to_var(c)
  character(*) :: c
  print *, c//c
end subroutine

! CHECK-LABEL: func.func @_QPexpr_to_var(
! CHECK:  %[[VAL_9:.*]] = hlfir.concat %{{.*}}, %{{.*}} len %[[VAL_8:.*]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:  %[[VAL_10:.*]]:3 = hlfir.associate %[[VAL_9]] typeparams %[[VAL_8]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_8]] : (index) -> i64
! CHECK:  %[[VAL_13:.*]] = fir.call @_FortranAioOutputAscii(%{{.*}}, %[[VAL_11]], %[[VAL_12]]) {{.*}} : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:  hlfir.end_associate %[[VAL_10]]#1, %[[VAL_10]]#2 : !fir.ref<!fir.char<1,?>>, i1
