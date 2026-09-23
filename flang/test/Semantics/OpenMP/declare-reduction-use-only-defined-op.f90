! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s

! Test that declare reduction for a defined operator works correctly with
! USE...ONLY when only the operator interface is imported.

module m_defined_op_reduction
  type :: dt2
    real :: x = 0.0
  end type
  interface operator(.combine.)
    module procedure combine_fn
  end interface
  !$omp declare reduction(.combine.:dt2:omp_out%x=omp_out%x+omp_in%x) &
  !$omp   initializer(omp_priv=dt2(0.0))
contains
  type(dt2) function combine_fn(a, b)
    type(dt2), intent(in) :: a, b
    combine_fn%x = a%x + b%x
  end function
end module

subroutine test_defined_op_use_only()
  use m_defined_op_reduction, only: dt2, operator(.combine.)
  type(dt2) :: y
  integer :: i
  y = dt2(0.0)
  ! Should compile without error: reduction is accessible via operator(.combine.)
  !$omp parallel do reduction(.combine.:y)
  do i = 1, 10
    y%x = y%x + 1.0
  end do
  !$omp end parallel do
end subroutine
