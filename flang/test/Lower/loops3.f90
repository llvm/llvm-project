! Test do concurrent reduction
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPloop_test
subroutine loop_test
  integer(4) :: i, j, k, tmp, sum = 0
  real :: m

  i = 100
  j = 200
  k = 300

  ! CHECK: %[[M:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFloop_testEm"}
  ! CHECK: %[[SUM:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFloop_testEsum"}
  ! CHECK: fir.do_concurrent.loop ({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) local(@_QFloop_testEtmp_private_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32>) reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[SUM]]#0 -> %{{.*}}, @max_reduction_f32 #fir.reduce_attr<max> %[[M]]#0 -> %{{.*}} : !fir.ref<i32>, !fir.ref<f32>) {
  ! CHECK: %[[TMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFloop_testEtmp"}
  ! CHECK: %[[SUM_INNER:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFloop_testEsum"}
  ! CHECK: %[[M_INNER:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFloop_testEm"}
  ! CHECK: hlfir.assign %{{.*}} to %[[TMP]]#0 : i32, !fir.ref<i32>
  ! CHECK: %[[TMPVAL:.*]] = fir.load %[[TMP]]#0 : !fir.ref<i32>
  ! CHECK: %[[SUMVAL:.*]] = fir.load %[[SUM_INNER]]#0 : !fir.ref<i32>
  ! CHECK: %[[ADDED:.*]] = arith.addi %[[TMPVAL]], %[[SUMVAL]] : i32
  ! CHECK: hlfir.assign %[[ADDED]] to %[[SUM_INNER]]#0 : i32, !fir.ref<i32>
  ! CHECK: hlfir.assign %{{.*}} to %[[M_INNER]]#0 : f32, !fir.ref<f32>
  do concurrent (i=1:5, j=1:5, k=1:5) local(tmp) reduce(+:sum) reduce(max:m)
    tmp = i + j + k
    sum = tmp + sum
    m = max(m, sum)
  enddo
end subroutine loop_test

! CHECK-LABEL: func.func @_QPloop_min_max_test
subroutine loop_min_max_test
  integer :: i
  real :: lo, hi
  lo = huge(0.0)
  hi = 0.0

  ! CHECK: fir.do_concurrent.loop
  ! CHECK-SAME: @min_reduction_f32 #fir.reduce_attr<min>
  ! CHECK-SAME: @max_reduction_f32 #fir.reduce_attr<max>
  do concurrent (i=1:10) reduce(min:lo) reduce(max:hi)
    lo = min(lo, real(i))
    hi = max(hi, real(i))
  enddo
end subroutine loop_min_max_test

! CHECK-LABEL: func.func @_QPloop_bitwise_test
subroutine loop_bitwise_test
  integer :: i
  integer :: a, o, x
  a = -1
  o = 0
  x = 0

  ! CHECK: fir.do_concurrent.loop
  ! CHECK-SAME: @iand_reduction_i32 #fir.reduce_attr<iand>
  ! CHECK-SAME: @ior_reduction_i32 #fir.reduce_attr<ior>
  ! CHECK-SAME: @ieor_reduction_i32 #fir.reduce_attr<ieor>
  do concurrent (i=1:10) reduce(iand:a) reduce(ior:o) reduce(ieor:x)
    a = iand(a, i)
    o = ior(o, i)
    x = ieor(x, i)
  enddo
end subroutine loop_bitwise_test
