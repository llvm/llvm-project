! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s

! Test that declare reduction resolves correctly when the same operator is
! imported (and renamed) from several modules, each declaring a reduction for a
! different type. The merged generic must resolve to the reduction matching the
! requested type, not just the first module's reduction.

module m_int
  type :: t_int
    integer :: val = 0
  end type
  interface operator(.remote.)
    module procedure add_int
  end interface
  !$omp declare reduction(.remote.:t_int:omp_out%val=omp_out%val+omp_in%val) &
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
  interface operator(.remote.)
    module procedure add_real
  end interface
  !$omp declare reduction(.remote.:t_real:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t_real(0.0))
contains
  type(t_real) function add_real(a, b)
    type(t_real), intent(in) :: a, b
    add_real%val = a%val + b%val
  end function
end module

program test_merged_use_only_reduction
  use m_int, only: t_int, operator(.local.) => operator(.remote.)
  use m_real, only: t_real, operator(.local.) => operator(.remote.)
  type(t_int) :: x
  type(t_real) :: y
  integer :: i
  x = t_int(0)
  y = t_real(0.0)
  ! The reduction for the first module (t_int) appears first in the merged
  ! generic. Resolving the second type (t_real) must still succeed.
  !$omp parallel do reduction(.local.:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
  !$omp parallel do reduction(.local.:y)
  do i = 1, 10
    y%val = y%val + 1.0
  end do
  !$omp end parallel do
end program
