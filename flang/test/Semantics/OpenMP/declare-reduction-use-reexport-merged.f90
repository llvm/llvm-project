! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s

! Test that declare reduction resolves through a re-exporting (facade) module.
! The intermediate module merges the same operator from two modules (each with
! a reduction for a different type) and re-exports it. A program that imports
! the facade must resolve the reduction for both types via the merged generic's
! USE associations, not only the operator directly in the facade.

module m_int
  type :: t_int
    integer :: val = 0
  end type
  interface operator(.shared.)
    module procedure add_int
  end interface
  !$omp declare reduction(.shared.:t_int:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t_int(0))
contains
  type(t_int) function add_int(a, b)
    type(t_int), intent(in) :: a, b
    add_int%val = a%val + b%val
  end function
end module

module m_real
  type :: t_real
    real :: val = 0.0
  end type
  interface operator(.shared.)
    module procedure add_real
  end interface
  !$omp declare reduction(.shared.:t_real:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t_real(0.0))
contains
  type(t_real) function add_real(a, b)
    type(t_real), intent(in) :: a, b
    add_real%val = a%val + b%val
  end function
end module

module m_facade
  use m_int, only: t_int, operator(.shared.)
  use m_real, only: t_real, operator(.shared.)
end module

program test_reexport_merged_reduction
  use m_facade
  type(t_int) :: x
  type(t_real) :: y
  integer :: i
  x = t_int(0)
  y = t_real(0.0)
  !$omp parallel do reduction(.shared.:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
  !$omp parallel do reduction(.shared.:y)
  do i = 1, 10
    y%val = y%val + 1.0
  end do
  !$omp end parallel do
end program
