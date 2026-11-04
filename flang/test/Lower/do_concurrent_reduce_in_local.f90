! Test a REDUCE locality-spec on a nested DO CONCURRENT naming a variable that
! is LOCAL or LOCAL_INIT in the enclosing DO CONCURRENT. The reduction must
! operate on the enclosing construct's copy of the variable, not on the
! variable of the subprogram.

! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -mmlir --enable-delayed-privatization=false \
! RUN:   -o - %s | FileCheck %s --check-prefix=NODELAY

subroutine reduce_in_local(n, nz, work)
  implicit none
  integer :: n, nz, work(nz), j, k, tmp
  do concurrent (k=1:nz) local(tmp)
    tmp = 0
    do concurrent (j=1:n) reduce(max:tmp)
      tmp = max(tmp, j)
    end do
    work(k) = tmp
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduce_in_local(
! CHECK:         %[[TMP:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFreduce_in_localEtmp")
! CHECK:         fir.do_concurrent.loop {{.*}} local(@{{.*}} %[[TMP]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFreduce_in_localEtmp")
! CHECK:           fir.do_concurrent.loop {{.*}} reduce(@max_reduction_i32 #fir.reduce_attr<max> %[[LOCAL]]#0 -> %[[RED_ARG:.*]] : !fir.ref<i32>) {
! CHECK:             %[[RED:.*]]:2 = hlfir.declare %[[RED_ARG]] uniq_name("_QFreduce_in_localEtmp")
! CHECK:             hlfir.assign %{{.*}} to %[[RED]]#0
! CHECK:           }
! CHECK:           fir.load %[[LOCAL]]#0

! NODELAY-LABEL: func.func @_QPreduce_in_local(
! NODELAY:         fir.do_concurrent.loop
! NODELAY:           %[[CLONE:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFreduce_in_localEtmp"
! NODELAY:           %[[LOCAL:.*]]:2 = hlfir.declare %[[CLONE]] uniq_name("_QFreduce_in_localEtmp")
! NODELAY:           fir.do_concurrent.loop {{.*}} reduce(@max_reduction_i32 #fir.reduce_attr<max> %[[LOCAL]]#0 -> %{{.*}} : !fir.ref<i32>) {

subroutine reduce_in_local_init(n, nz, work)
  implicit none
  integer :: n, nz, work(nz), j, k, tmp
  tmp = 0
  do concurrent (k=1:nz) local_init(tmp)
    do concurrent (j=1:n) reduce(+:tmp)
      tmp = tmp + j
    end do
    work(k) = tmp
  end do
end subroutine

! CHECK-LABEL: func.func @_QPreduce_in_local_init(
! CHECK:         %[[TMP:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFreduce_in_local_initEtmp")
! CHECK:         fir.do_concurrent.loop {{.*}} local(@{{.*}} %[[TMP]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<i32>) {
! CHECK:           %[[LOCAL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] uniq_name("_QFreduce_in_local_initEtmp")
! CHECK:           fir.do_concurrent.loop {{.*}} reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[LOCAL]]#0 -> %{{.*}} : !fir.ref<i32>) {

! NODELAY-LABEL: func.func @_QPreduce_in_local_init(
! NODELAY:         fir.do_concurrent.loop
! NODELAY:           %[[CLONE:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFreduce_in_local_initEtmp"
! NODELAY:           %[[LOCAL:.*]]:2 = hlfir.declare %[[CLONE]] uniq_name("_QFreduce_in_local_initEtmp")
! NODELAY:           fir.do_concurrent.loop {{.*}} reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[LOCAL]]#0 -> %{{.*}} : !fir.ref<i32>) {
