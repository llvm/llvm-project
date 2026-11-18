! RUN: %python %S/test_errors.py %s %flang_fc1

! The two specifics are distinguishable by the type of the function that is the
! dummy procedure f. An external procedure with an implicit interface matches
! both, and its type is not known, so the call is ambiguous; a procedure with an
! explicit interface selects one.
module implicit_interface
  type :: t
  contains
    procedure :: a => sa
    procedure, nopass :: b => sb
    generic :: g => a, b
  end type
contains
  subroutine sa(x, f)
    class(t) :: x
    interface
      real function f()
      end function
    end interface
  end subroutine
  subroutine sb(f)
    interface
      integer function f()
      end function
    end interface
  end subroutine
  real function real_result()
    real_result = 1.
  end function
  integer function integer_result()
    integer_result = 1
  end function
  subroutine user(v)
    type(t) :: v
    external ext
    !ERROR: The actual arguments to the generic procedure 'g' matched multiple specific procedures, perhaps due to use of NULL() without MOLD= or an actual procedure with an implicit interface
    call v%g(ext)
    call v%g(real_result)
    call v%g(integer_result)
  end subroutine
end module
