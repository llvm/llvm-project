!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=45 -Wno-openmp-future -Werror -Wno-experimental-option -o - %s | FileCheck %s

! IF clause is not allowed on SIMD in OpenMP 4.5, but is allowed in a later
! version. We emit a warning for that, and after the semantic checks the
! clause should be treated as allowed. Make sure that this is still the case
! when the diagnostic is suppressed and -Werror is present.

! This code should compile successfully.

! CHECK: omp.simd if(%true)

subroutine leaf(a, n)
  integer :: n, i
  real :: a(n)
  !$omp simd if(.true.)
  do i = 1, n
    a(i) = a(i) + 1.0
  end do
  !$omp end simd
end subroutine
