! RUN: bbc -fopenacc -pft-test -o %t %s | FileCheck %s
! RUN: %flang_fc1 -fopenacc -fdebug-dump-pft -o %t %s | FileCheck %s

! Test structure of the Pre-FIR tree with OpenACC

subroutine sub1(a, b, n)
  real :: a(:), b(:)
  integer :: n, i
  !$acc parallel loop present(a, b)
  do i = 1, n
    b(i) = exp(a(i))
  end do
end subroutine

! CHECK-LABEL: Subroutine sub1
! CHECK:       <<OpenACCConstruct>>
! CHECK:       <<DoConstruct>>
! CHECK:       <<End OpenACCConstruct>>
