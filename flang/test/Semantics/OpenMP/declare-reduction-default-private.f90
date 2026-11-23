! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 %s 2>&1 | FileCheck %s

! Test that a module with default PRIVATE accessibility propagates
! the PRIVATE attribute to declare reduction symbols.

module m_default_private
  private
  type, public :: dt
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

program test_default_private
  use m_default_private
  type(dt) :: x
  integer :: i
  x = dt(0)
  ! The reduction should be PRIVATE because the module default is PRIVATE
  ! and operator(+) has no explicit PUBLIC.
  !CHECK: error: The type of 'x' is incompatible with the reduction operator.
  !$omp parallel do reduction(+:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
end program
