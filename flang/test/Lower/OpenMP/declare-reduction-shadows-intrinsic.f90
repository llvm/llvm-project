!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Test that a user-defined reduction whose identifier is the same as an
! intrinsic reduction (max/min/iand/ior/ieor) shadows the intrinsic: the
! reduction clause must bind to the user-declared reduction, not to the
! built-in one.

subroutine test_max(x)
  integer :: x(10), r, i
!$omp declare reduction(max:integer:omp_out=omp_out+omp_in) initializer(omp_priv=0)
  r = 0
!$omp parallel do reduction(max:r)
  do i=1,10
     r = r + x(i)
  end do
end subroutine

! The user-defined reduction must be materialized and its combiner must be the
! user's addition (not the intrinsic max's select/compare). No intrinsic
! @max_reduction / @max_i32 op should be generated.

! CHECK-NOT: omp.declare_reduction @max
! CHECK: omp.declare_reduction @[[RED:_QQFtest_maxop\.max_i32]] : i32 init {
! CHECK: combiner {
! CHECK: arith.addi
! CHECK: omp.yield

! CHECK-LABEL: func.func @_QPtest_max
! CHECK: omp.wsloop {{.*}}reduction(@[[RED]] %{{.*}} -> %{{.*}} : !fir.ref<i32>)
! CHECK-NOT: omp.declare_reduction @max
