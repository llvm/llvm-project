! RUN: bbc -fopenmp -pft-test -o %t %s | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fdebug-dump-pft -o %t %s | FileCheck %s

! Test structure of the Pre-FIR tree with OpenMP

subroutine sub1(a, b, n)
  real :: a(:), b(:)
  integer :: n, i
  !$omp parallel do
  do i = 1, n
    b(i) = exp(a(i))
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: Subroutine sub1
! CHECK:       <<OpenMPConstruct>>
! CHECK:       <<DoConstruct>>
! CHECK:       <<End OpenMPConstruct>>
