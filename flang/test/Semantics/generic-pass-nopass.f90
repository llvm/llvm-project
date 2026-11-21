! RUN: %python %S/test_errors.py %s %flang_fc1

! F2023 15.4.3.4.5 defines effective positions after removing the passed
! object. Named generics may mix PASS and NOPASS, but the extra implicit
! argument does not distinguish otherwise identical interfaces.
module mixed_bindings
  type :: t
  contains
    procedure, pass(this) :: with_object
    procedure, nopass :: without_object
    procedure, nopass :: real_argument
    ! Preserve the declaration diagnostics as well as diagnosing the calls.
    !ERROR: Generic 'ambiguous' may not have specific procedures 'with_object' and 'without_object' as their interfaces are not distinguishable
    generic :: ambiguous => with_object, without_object
    !ERROR: Generic 'reversed' may not have specific procedures 'with_object' and 'without_object' as their interfaces are not distinguishable
    generic :: reversed => without_object, with_object
    generic :: valid => with_object, real_argument
  end type
contains
  subroutine with_object(x, this)
    integer :: x
    class(t) :: this
  end subroutine
  subroutine without_object(x)
    integer :: x
  end subroutine
  subroutine real_argument(x)
    real :: x
  end subroutine
  subroutine caller(obj, x)
    type(t) :: obj
    integer :: x
    ! Both of these calls must be diagnosed, rather than comparing vectors
    ! with different lengths and aborting. PASS(this) need not be first.
    !ERROR: The actual arguments to the generic procedure 'ambiguous' matched multiple specific procedures, perhaps due to use of NULL() without MOLD= or an actual procedure with an implicit interface
    call obj%ambiguous(x)
    !ERROR: The actual arguments to the generic procedure 'reversed' matched multiple specific procedures, perhaps due to use of NULL() without MOLD= or an actual procedure with an implicit interface
    call obj%reversed(x=x)
    ! A standard-conforming mixed generic remains resolvable by its explicit
    ! argument, both positionally and by keyword.
    call obj%valid(x)
    call obj%valid(x=1.0)
  end subroutine
end module
