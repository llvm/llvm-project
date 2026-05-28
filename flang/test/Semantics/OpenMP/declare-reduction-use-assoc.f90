! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fdebug-dump-symbols %s 2>&1 | FileCheck %s

! Test that USE-associated declare reduction does not produce false
! "Duplicate definition" errors when a new local declaration is made.
! Related: https://github.com/llvm/llvm-project/issues/192580

module m_use_reduction
  type :: dt
    integer :: val = 0
  end type
  !$omp declare reduction(+:dt:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=dt(0))
end module

module m_local_reduction
  use m_use_reduction, only: dt
  type :: dt2
    real :: x = 0.0
  end type
  ! This should NOT produce "Duplicate definition" — the USE-associated
  ! reduction for dt is shadowed by this new local declaration for dt2.
  !$omp declare reduction(+:dt2:omp_out%x=omp_out%x+omp_in%x) &
  !$omp   initializer(omp_priv=dt2(0.0))
end module

!CHECK: Module scope: m_local_reduction
!CHECK: op.+, PUBLIC: UserReductionDetails TYPE(dt2)
