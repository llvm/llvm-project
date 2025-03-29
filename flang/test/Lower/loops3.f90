! Test do concurrent reduction
! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: loop_test
subroutine loop_test
  integer(4) :: i, j, k, tmp, sum = 0
  real :: m

  i = 100
  j = 200
  k = 300

  ! CHECK: %[[VAL_0:.*]] = fir.alloca f32 {bindc_name = "m", uniq_name = "_QFloop_testEm"}
  ! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFloop_testEsum) : !fir.ref<i32>
  ! CHECK: fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered reduce(#fir.reduce_attr<add> -> %[[VAL_1:.*]] : !fir.ref<i32>, #fir.reduce_attr<max> -> %[[VAL_0:.*]] : !fir.ref<f32>) {
  ! CHECK: fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered reduce(#fir.reduce_attr<add> -> %[[VAL_1:.*]] : !fir.ref<i32>, #fir.reduce_attr<max> -> %[[VAL_0:.*]] : !fir.ref<f32>) {
  ! CHECK: fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered reduce(#fir.reduce_attr<add> -> %[[VAL_1:.*]] : !fir.ref<i32>, #fir.reduce_attr<max> -> %[[VAL_0:.*]] : !fir.ref<f32>) {
  do concurrent (i=1:5, j=1:5, k=1:5) local(tmp) reduce(+:sum) reduce(max:m)
    tmp = i + j + k
    sum = tmp + sum
    m = max(m, sum)
  enddo
end subroutine loop_test
