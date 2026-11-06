!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Test lowering of a user-defined reduction on a LOGICAL variable whose
! initializer is a logical literal (omp_priv = .false.). The literal lowers to
! an i1, but the reduction type is !fir.logical<4>; the init region must yield
! the reduction type.

subroutine test_logical(r)
  logical :: r
  integer :: i
!$omp declare reduction(my_and:logical:omp_out=omp_in.and.omp_out) initializer(omp_priv=.false.)
!$omp parallel do reduction(my_and:r)
  do i=1,2
  end do
end subroutine

! CHECK: omp.declare_reduction @[[RED:_QQFtest_logicalmy_and_l32]] : !fir.logical<4> init {
! CHECK: %[[FALSE:.*]] = arith.constant false
! CHECK: %[[CONV:.*]] = fir.convert %[[FALSE]] : (i1) -> !fir.logical<4>
! CHECK: omp.yield(%[[CONV]] : !fir.logical<4>)
! CHECK: } combiner {
! CHECK: fir.logical_and
! CHECK: omp.yield(%{{.*}} : !fir.logical<4>)

! CHECK-LABEL: func.func @_QPtest_logical
! CHECK: omp.wsloop {{.*}}reduction(@[[RED]] %{{.*}} -> %{{.*}} : !fir.ref<!fir.logical<4>>)
