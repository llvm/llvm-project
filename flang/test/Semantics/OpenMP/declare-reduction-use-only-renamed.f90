! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fdebug-dump-symbols %s | FileCheck %s

! Test that declare reduction works correctly with USE...ONLY when
! the operator is renamed during import.

module m_remote_reduction
  type :: t
    integer :: val = 0
  end type
  interface operator(.remote.)
    module procedure add_t
  end interface
  !$omp declare reduction(.remote.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
! CHECK: op.remote., PUBLIC: UserReductionDetails TYPE(t)
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

program test_renamed_use_only_reduction
  use m_remote_reduction, only: t, operator(.local.) => operator(.remote.)
! CHECK: .local. (Function): Use from .remote. in m_remote_reduction
  type(t) :: x
  integer :: i
  x = t(0)
  ! Should compile without error: reduction is accessible via renamed operator
  !$omp parallel do reduction(.local.:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
end program
