! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 %s 2>&1 | FileCheck %s

! Test that PRIVATE operator accessibility is propagated to declare reduction
! and enforced when the reduction is used from outside the module.
! Related: https://github.com/llvm/llvm-project/issues/187415

module m_private_op
  type :: dt
    integer :: val = 0
  end type
  private :: operator(+)
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

program test_private_reduction
  use m_private_op
  type(dt) :: x
  integer :: i
  x = dt(0)
  !CHECK: error: The type of 'x' is incompatible with the reduction operator.
  !$omp parallel do reduction(+:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
end program
