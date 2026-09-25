! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=UNPARSE
! Test semantics of F2023 TYPEOF and CLASSOF type specifiers.

module m
  type :: base_type
    integer :: x
  end type
  type, extends(base_type) :: child_type
    integer :: y
  end type
contains
  subroutine test_typeof_derived(a, b)
    type(base_type) :: a
    type(child_type) :: b
    !UNPARSE: TYPEOF(a) :: c
    typeof(a) :: c
    !UNPARSE: TYPEOF(b) :: d
    typeof(b) :: d
  end subroutine

  subroutine test_typeof_intrinsic(a, b, c)
    integer :: a
    real(8) :: b
    logical :: c
    !UNPARSE: TYPEOF(a) :: d
    typeof(a) :: d
    !UNPARSE: TYPEOF(b) :: e
    typeof(b) :: e
    !UNPARSE: TYPEOF(c) :: f
    typeof(c) :: f
  end subroutine

  subroutine test_typeof_assumed_char(a)
    character(*) :: a
    !UNPARSE: TYPEOF(a) :: b
    typeof(a) :: b
  end subroutine

  subroutine test_classof(a)
    class(base_type), intent(in) :: a
    !UNPARSE: CLASSOF(a), ALLOCATABLE :: b
    classof(a), allocatable :: b
  end subroutine

  subroutine test_typeof_assumed_type(a, b)
    type(*), intent(in) :: a
    !UNPARSE: TYPEOF(a) :: b
    typeof(a) :: b
  end subroutine

  subroutine test_classof_unlimited_poly(a)
    class(*), intent(in) :: a
    !UNPARSE: CLASSOF(a), ALLOCATABLE :: b
    classof(a), allocatable :: b
  end subroutine
end module
