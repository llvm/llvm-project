!RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
module m
  type :: t1
    integer :: n
   contains
    final :: t1f0, t1f1
  end type
  type :: t2
    integer :: n
   contains
    final :: t2fe
  end type
  type :: t3
    integer :: n
   contains
    final :: t3far
  end type
  type, extends(t1) :: t4
  end type
  type :: t5
    !CHECK-NOT: 'scalar' of derived type 't1'
    type(t1) :: scalar
    !CHECK-NOT: 'vector' of derived type 't1'
    type(t1) :: vector(2)
    !CHECK: 'matrix' of derived type 't1' does not have a FINAL subroutine for its rank (2)
    type(t1) :: matrix(2, 2)
  end type
 contains
  subroutine t1f0(x)
    type(t1) :: x
  end subroutine
  subroutine t1f1(x)
    type(t1) :: x(:)
  end subroutine
  impure elemental subroutine t2fe(x)
    type(t2), intent(in out) :: x
  end subroutine
  subroutine t3far(x)
    type(t3) :: x(..)
  end subroutine
end module

subroutine test ! *not* a main program, since they don't finalize locals
  use m
  !CHECK-NOT: 'scalar1' of derived type 't1'
  type(t1) :: scalar1
  !CHECK-NOT: 'vector1' of derived type 't1'
  type(t1) :: vector1(2)
  !CHECK: 'matrix1' of derived type 't1' does not have a FINAL subroutine for its rank (2)
  type(t1) :: matrix1(2,2)
  !CHECK-NOT: 'scalar2' of derived type 't2'
  type(t2) :: scalar2
  !CHECK-NOT: 'vector2' of derived type 't2'
  type(t2) :: vector2(2)
  !CHECK-NOT: 'matrix2' of derived type 't2'
  type(t2) :: matrix2(2,2)
  !CHECK-NOT: 'scalar3' of derived type 't3'
  type(t3) :: scalar3
  !CHECK-NOT: 'vector3' of derived type 't3'
  type(t3) :: vector3(2)
  !CHECK-NOT: 'matrix3' of derived type 't2'
  type(t3) :: matrix3(2,2)
  !CHECK-NOT: 'scalar4' of derived type 't4'
  type(t4) :: scalar4
  !CHECK-NOT: 'vector4' of derived type 't4'
  type(t4) :: vector4(2)
  !CHECK: 'matrix4' of derived type 't4' extended from 't1' does not have a FINAL subroutine for its rank (2)
  type(t4) :: matrix4(2,2)
end
