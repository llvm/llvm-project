! A user-defined operator reduction whose declaration is USE-associated from a
! module (plain USE, so the "op<spelling>" reduction symbol is imported) reaches
! lowering but is not yet supported: lowering does not materialize imported
! declare reductions. It must emit a clean TODO rather than ICE (#204299).

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP user-defined operator reduction is not yet supported for imported, renamed, or multiple-declaration/type reductions

module m_use_op
  type :: t
    integer :: val = 0
  end type
  interface operator(.plus.)
    module procedure add_t
  end interface
  !$omp declare reduction(.plus.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function add_t
end module m_use_op

program p
  use m_use_op
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.plus.:x)
  do i = 1, 100
    x = x .plus. t(1)
  end do
  !$omp end parallel do
end program p
