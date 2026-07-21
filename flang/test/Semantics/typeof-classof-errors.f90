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

  subroutine test_typeof_unlimited_poly(a)
    class(*), intent(in) :: a
    !ERROR: The data-ref in TYPEOF must not be unlimited polymorphic
    typeof(a) :: b
  end subroutine

  subroutine test_classof_assumed_type(a)
    type(*), intent(in) :: a
    !ERROR: The data-ref in CLASSOF must not be assumed-type
    classof(a) :: b
  end subroutine

  subroutine test_typeof_optional_assumed_char(a)
    character(*), optional :: a
    !ERROR: The OPTIONAL data-ref in TYPEOF must not have assumed or deferred type parameters
    typeof(a) :: b
  end subroutine

  subroutine test_typeof_optional_deferred_char(a)
    character(:), allocatable, optional :: a
    !ERROR: The OPTIONAL data-ref in TYPEOF must not have assumed or deferred type parameters
    typeof(a) :: b
  end subroutine

  subroutine test_typeof_optional_ok(a)
    integer, optional :: a
    typeof(a) :: b
  end subroutine

  subroutine test_typeof_deferred_char_local(c)
    character(:), allocatable :: c
    !ERROR: 'localc' has a type CHARACTER(KIND=1,LEN=:) with a deferred type parameter but is neither an allocatable nor an object pointer
    typeof(c) :: localc
  end subroutine
end module
