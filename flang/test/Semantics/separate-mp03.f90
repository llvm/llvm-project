! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests module procedures declared and defined in the same module.

! These cases are correct.
module m1
  interface
    integer module function f1(x)
      real, intent(in) :: x
    end function
    integer module function f2(x)
      real, intent(in) :: x
    end function
    module function f3(x) result(res)
      integer :: res
      real, intent(in) :: x
    end function
    module function f4(x) result(res)
      integer :: res
      real, intent(in) :: x
    end function
    module subroutine s1
    end subroutine
    pure module subroutine s2
    end subroutine
    module subroutine s3
    end subroutine
  end interface
 contains
  integer module function f1(x)
    real, intent(in) :: x
    f1 = x
  end function
  module procedure f2
    f2 = x
  end procedure
  module function f3(x) result(res)
    integer :: res
    real, intent(in) :: x
    res = x
  end function
  module procedure f4
    res = x
  end procedure
  module subroutine s1
  end subroutine
  pure module subroutine s2
  end subroutine
  module procedure s3
  end procedure
end module

! Error cases

module m2
  interface
    integer module function f1(x)
      real, intent(in) :: x
    end function
    integer module function f2(x)
      real, intent(in) :: x
    end function
    module function f3(x) result(res)
      integer :: res
      real, intent(in) :: x
    end function
    module function f4(x) result(res)
      integer :: res
      real, intent(in) :: x
    end function
    module subroutine s1
    end subroutine
    pure module subroutine s2
    end subroutine
  end interface
 contains
  integer module function f1(x)
    !ERROR: Dummy argument 'x' has type INTEGER(4); the corresponding argument in the interface body has distinct type REAL(4)
    integer, intent(in) :: x
    f1 = x
  end function
  !ERROR: 'notf2' was not declared a separate module procedure
  module procedure notf2
  end procedure
  !ERROR: Result of function 'f3' is not compatible with the result of the corresponding interface body: function results have distinct types: REAL(4) vs INTEGER(4)
  module function f3(x) result(res)
    real :: res
    real, intent(in) :: x
    res = x
  end function
  !ERROR: Module subroutine 'f4' was declared as a function in the corresponding interface body
  module subroutine f4
  end subroutine
  !ERROR: Module function 's1' was declared as a subroutine in the corresponding interface body
  module function s1
  end function
  !ERROR: Module subprogram 's2' and its corresponding interface body are not both PURE
  impure module subroutine s2
  end subroutine
end module
