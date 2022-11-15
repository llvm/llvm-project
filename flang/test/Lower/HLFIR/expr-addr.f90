! Test lowering of of expressions as address
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPfoo(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>
subroutine foo(x)
  integer :: x
  read (*,*) x
  ! CHECK: %[[x:.]]:2 = hlfir.declare %[[arg0]] {uniq_name = "_QFfooEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[x_cast:.*]] = fir.convert %[[x]]#1 : (!fir.ref<i32>) -> !fir.ref<i64>
  ! CHECK: fir.call @_FortranAioInputInteger(%{{.*}}, %[[x_cast]], %{{.*}}) : (!fir.ref<i8>, !fir.ref<i64>, i32) -> i1
end subroutine
