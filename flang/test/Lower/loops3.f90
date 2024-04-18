! Test do concurrent reduction
! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: loop_test
subroutine loop_test
  integer(4) :: i, j, k, tmp, sum = 0
  real :: m

  i = 100
  j = 200
  k = 300

  ! CHECK: %[[VAL_0:.*]] = fir.reduce %{{.*}} {name = "sum"} : (!fir.ref<i32>) -> !fir.ref<i32>
  ! CHECK: %[[VAL_1:.*]] = fir.reduce %{{.*}} {name = "m"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK: fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered reduce(#fir.reduce_attr<add> -> %[[VAL_0]] : !fir.ref<i32>, #fir.reduce_attr<max> -> %[[VAL_1]] : !fir.ref<f32>) attributes {operandSegmentSizes = array<i32: 1, 1, 1, 0, 2>} {
  do concurrent (i=1:5, j=1:5, k=1:5) local(tmp) reduce(+:sum) reduce(max:m)
    tmp = i + j + k
    sum = tmp + sum
    m = max(m, sum)
  enddo
end subroutine loop_test
