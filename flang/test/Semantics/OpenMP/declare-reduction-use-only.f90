! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s

! Test that declare reduction works correctly with USE...ONLY when
! only the operator (not the internal reduction symbol) is imported.

module m_with_reduction
  type :: dt
    integer :: val = 0
  end type
  interface operator(+)
    module procedure add_dt
  end interface
  !$omp declare reduction(+:dt:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=dt(0))
contains
  type(dt) function add_dt(a, b)
    type(dt), intent(in) :: a, b
    add_dt%val = a%val + b%val
  end function
end module

! USE...ONLY imports operator(+) but not the internal op.+ symbol
program test_use_only
  use m_with_reduction, only: dt, operator(+)
  type(dt) :: x
  integer :: i
  x = dt(0)
  ! Should compile without error: reduction is accessible via operator(+)
  !$omp parallel do reduction(+:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
  print *, x%val
end program
