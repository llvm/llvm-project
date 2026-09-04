! Test lowering of `lastprivate(conditional:)` on a worksharing do that is
! lexically nested inside TWO parallel regions (nested parallelism, non-orphaned
! / "inlined").  Because an enclosing omp.parallel exists in the same function,
! the reduction struct is a per-thread stack alloca placed before the innermost
! enclosing parallel, so each outer thread gets its own copy.  No runtime
! nested-parallelism guard is emitted here -- that guard is only needed for the
! orphaned case, which shares a module-scope global (see
! lastprivate-conditional-wsloop-orphaned.f90).

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp_nested_parallel(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i

  !$omp parallel
  !$omp parallel
  !$omp do lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 3) == 0) x = i
  end do
  !$omp end do
  !$omp end parallel
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_conditional_lp_nested_parallel
! No orphaned-case runtime guard for the inlined case: no omp_get_level check
! and no ERROR STOP before the region.
! CHECK-NOT:     omp_get_level
! CHECK-NOT:     _FortranAStopStatementText
! Outer parallel; the reduction struct is a per-thread alloca created inside it
! (before the inner parallel), so each outer thread has its own copy.
! CHECK:         omp.parallel {
! CHECK:           %[[S:.*]] = fir.alloca !fir.type<_lp_cond_t{{.*}}> {pinned}
! CHECK:           omp.parallel {
! The inner worksharing loop reduces into that per-outer-thread struct.
! CHECK:             omp.wsloop
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK:             omp.single {
! CHECK:               arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:               omp.terminator
! CHECK:             }
