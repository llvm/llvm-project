! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 %s 2>&1 | FileCheck %s

! Test that a declare reduction from a module that was not USE'd (or only
! partially USE'd) is not incorrectly found during type checking.
! Related: https://github.com/llvm/llvm-project/issues/200300

module m_with_reduction
  type :: t
    integer :: val = 0
  end type
  !$omp declare reduction(+:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
end module

! proxy re-exports only the type, not the reduction
module m_proxy
  use m_with_reduction, only: t
end module

program test_overbroad_lookup
  use m_proxy
  type(t) :: x
  integer :: i
  x = t(0)
  !CHECK: error: The type of 'x' is incompatible with the reduction operator.
  !$omp parallel do reduction(+:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
  print *, x%val
end program
