! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=45

! The ALLOCATE clause was introduced in OpenMP 5.0, so before that it must be
! rejected rather than accepted and handed to lowering, which cannot decompose
! it -- see llvm-project#211430.
!
! Every directive below was previously ungated, i.e. it accepted the clause at
! any version. `do`, `parallel do`, `taskgroup` and `scope` are deliberately
! not covered here: they were already gated before this change, so they would
! pass with or without it and would not exercise the new gate.
!
! The version is pinned rather than left to default so the expected message
! stays stable if the default OpenMP version changes.

subroutine allocate_pre50(n, a, b)
  integer :: n, i
  real :: a(n), b(n)
  real :: t

  !ERROR: ALLOCATE clause is not allowed on PARALLEL directive in OpenMP v4.5, try -fopenmp-version=50
  !$omp parallel private(t) allocate(t)
  t = a(1)
  b(1) = t
  !$omp end parallel

  !ERROR: ALLOCATE clause is not allowed on SINGLE directive in OpenMP v4.5, try -fopenmp-version=50
  !$omp single private(t) allocate(t)
  t = a(1)
  b(1) = t
  !$omp end single

  !ERROR: ALLOCATE clause is not allowed on TASK directive in OpenMP v4.5, try -fopenmp-version=50
  !$omp task private(t) allocate(t)
  t = a(1)
  b(1) = t
  !$omp end task

  ! The form reported in llvm-project#211430, where the ungated clause reached
  ! lowering and crashed it.
  !ERROR: ALLOCATE clause is not allowed on TARGET TEAMS DISTRIBUTE PARALLEL DO directive in OpenMP v4.5, try -fopenmp-version=50
  !$omp target teams distribute parallel do private(t) allocate(t)
  do i = 1, n
     t = a(i)
     b(i) = t
  end do
  !$omp end target teams distribute parallel do
end subroutine
