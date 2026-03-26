! RUN: %python %S/test_errors.py %s %flang_fc1
! Test semantic errors for F2023 TYPEOF and CLASSOF type specifiers.

module m
  type :: base_type
    integer :: x
  end type
  type :: non_extensible_type
    sequence
    integer :: x
  end type
contains
  subroutine test_typeof_subscript(a)
    integer :: a(10)
    !ERROR: The data-ref in TYPEOF must not have subscripts
    typeof(a(1)) :: b
  end subroutine

  subroutine test_classof_intrinsic(a)
    integer :: a
    !ERROR: CLASSOF may not be used with an intrinsic-type object
    classof(a) :: b
  end subroutine

  subroutine test_typeof_not_found()
    implicit none
    !ERROR: No explicit type declared for 'nonexistent'
    !ERROR: No explicit type declared for 'b'
    typeof(nonexistent) :: b
  end subroutine

  subroutine test_classof_non_extensible(a)
    type(non_extensible_type) :: a
    !ERROR: CLASSOF requires a data-ref of extensible type
    classof(a) :: b
  end subroutine

  subroutine test_typeof_assumed_size(a)
    integer :: a(*)
    !ERROR: The data-ref in TYPEOF must not be a whole assumed-size array
    typeof(a) :: b
  end subroutine
end module
