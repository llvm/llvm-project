!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=45 -o - %s 2>&1 | FileCheck %s

! Check that if we recommend using version 5.0 then we don't mark the IF
! clause as allowed on leafs that only allow it on versions later than 5.0.
! In this case SIMD allows IF in 5.0+, but TEAMS only allows it in 5.2.

! CHECK: warning: IF clause is not allowed on TEAMS DISTRIBUTE SIMD directive in OpenMP v4.5, try -fopenmp-version=50
! CHECK: omp.teams {
! CHECK: omp.simd if(%false)

subroutine tds(a, n)
  integer :: n, i
  real :: a(n)
  !$omp target map(tofrom: a)
  !$omp teams distribute simd if(.false.)
  do i = 1, n
    a(i) = a(i) + 1.0
  end do
  !$omp end teams distribute simd
  !$omp end target
end subroutine

