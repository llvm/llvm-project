! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fdebug-dump-symbols %s 2>&1 | FileCheck %s

! Test that PUBLIC operator accessibility still allows declare reduction
! to be used from outside the module (regression test).

module m_public_op
  type :: dt
    integer :: val = 0
  end type
  public :: operator(+)
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

!CHECK: Module scope: m_public_op
!CHECK: op.+, PUBLIC: UserReductionDetails TYPE(dt)

program test_public_reduction
  use m_public_op
  type(dt) :: x
  integer :: i
  x = dt(0)
  ! No error expected: operator(+) is PUBLIC so reduction is accessible.
  !$omp parallel do reduction(+:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
  print *, x%val
end program
