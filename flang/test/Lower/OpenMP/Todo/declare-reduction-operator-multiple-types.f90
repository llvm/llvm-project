! A user-defined operator declare reduction listing multiple types reaches
! lowering but is not yet supported (the op name is not type-specific, so the
! per-type ops would collide). It must emit a clean TODO rather than ICE or
! miscompile (#204299). The guard is on the directive side, which is lowered
! before any reduction clause.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP user-defined operator declare reduction with multiple declarations or multiple types

program p
  type :: t1
    integer :: v = 0
  end type
  type :: t2
    integer :: w = 0
  end type
  interface operator(.mt.)
    function f1(a, b)
      import :: t1
      type(t1), intent(in) :: a, b
      type(t1) :: f1
    end function f1
    function f2(a, b)
      import :: t2
      type(t2), intent(in) :: a, b
      type(t2) :: f2
    end function f2
  end interface
  !$omp declare reduction(.mt.:t1,t2:omp_out=omp_in)
  type(t1) :: x1
  integer :: i
  !$omp parallel do reduction(.mt.:x1)
  do i = 1, 5
    x1 = x1 .mt. t1(1)
  end do
  !$omp end parallel do
end program p
