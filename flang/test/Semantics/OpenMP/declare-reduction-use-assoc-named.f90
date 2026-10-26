! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fdebug-dump-symbols %s 2>&1 | FileCheck %s

! Test that USE-associated named reductions and user-defined operator
! reductions are correctly resolved through UseDetails.

module m_named_reduction
  type :: t
    integer :: val = 0
  end type
  !$omp declare reduction(myred:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
end module

module m_defined_op_reduction
  type :: dt
    real :: x = 0.0
  end type
  interface operator(.combine.)
    module procedure combine_fn
  end interface
  !$omp declare reduction(.combine.:dt:omp_out%x=omp_out%x+omp_in%x) &
  !$omp   initializer(omp_priv=dt(0.0))
contains
  type(dt) function combine_fn(a, b)
    type(dt), intent(in) :: a, b
    combine_fn%x = a%x + b%x
  end function
end module

program test_use_assoc_reductions
  use m_named_reduction
  use m_defined_op_reduction
  type(t) :: x
  type(dt) :: y
  integer :: i
  x = t(0)
  y = dt(0.0)
  ! Both should compile without error: reductions are accessible via USE.
  !$omp parallel do reduction(myred:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
  !$omp parallel do reduction(.combine.:y)
  do i = 1, 10
    y%x = y%x + 1.0
  end do
  !$omp end parallel do
  print *, x%val, y%x
end program

! Test defined operator with external interface via USE (issue #184932 pattern).
! Uses !$omp parallel (not parallel do) to cover that variant.
module m_external_op
  type :: ty
    integer :: ii
  end type
  interface operator(.x.)
    function h(a, b)
      import :: ty
      type(ty), intent(in) :: a, b
    end function
  end interface
  !$omp declare reduction(.x.:ty:omp_out=ty(1)) initializer(omp_priv=ty(0))
end module

subroutine test_external_op_reduction
  use m_external_op
  type(ty) :: v
  v = ty(0)
  !$omp parallel reduction(.x.:v)
  v = ty(1)
  !$omp end parallel
end subroutine

!CHECK: Module scope: m_named_reduction
!CHECK: myred, PUBLIC: UserReductionDetails TYPE(t)
!CHECK: Module scope: m_defined_op_reduction
!CHECK: op.combine., PUBLIC: UserReductionDetails TYPE(dt)
!CHECK: Module scope: m_external_op
!CHECK: op.x., PUBLIC: UserReductionDetails TYPE(ty)
