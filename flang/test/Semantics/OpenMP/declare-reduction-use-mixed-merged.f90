! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s

! Test that declare reduction resolves the type-correct reduction when an
! operator carries reductions for different types and one module is imported
! with a bare USE (making its reduction directly visible) while another is
! imported with USE...ONLY. The directly visible reduction must not shadow the
! reduction for the other type.

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

program test_mixed_merged_reduction
  use m_int                                     ! bare USE: t_int reduction directly visible
  use m_real, only: t_real, operator(.shared.)  ! USE...ONLY: operator merged in
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
  ! The t_int reduction from m_int is directly visible, but resolving t_real
  ! must still find m_real's reduction rather than stopping at t_int.
  !$omp parallel do reduction(.shared.:y)
  do i = 1, 10
    y%val = y%val + 1.0
  end do
  !$omp end parallel do
end program
