! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type :: pdt(len)
    integer, len :: len
    character(len=len) :: ch
  end type
 contains
  impure real function f1(x,y)
    f1 = x + y
  end function
  pure function f2(x,y)
    real :: f2(1)
    real, intent(in) :: x, y
    f2(1) = x + y
  end function
  pure real function f3(x,y,z)
    real, intent(in) :: x, y, z
    f3 = x + y + z
  end function
  pure real function f4(x,y)
    interface
      pure real function x(); end function
      pure real function y(); end function
    end interface
    f4 = x() + y()
  end function
  pure integer function f5(x,y)
    real, intent(in) :: x, y
    f5 = x + y
  end function
  pure real function f6(x,y)
    real, intent(in) :: x(*), y(*)
    f6 = x(1) + y(1)
  end function
  pure real function f7(x,y)
    real, intent(in), allocatable :: x
    real, intent(in) :: y
    f7 = x + y
  end function
  pure real function f8(x,y)
    real, intent(in), pointer :: x
    real, intent(in) :: y
    f8 = x + y
  end function
  pure real function f9(x,y)
    real, intent(in), optional :: x
    real, intent(in) :: y
    f9 = x + y
  end function
  pure real function f10(x,y)
    real, intent(in), target :: x
    real, intent(in) :: y
    f10 = x + y
  end function
  pure function f11(x,y) result(res)
    type(pdt(*)), intent(in) :: x, y
    type(pdt(max(x%len, y%len))) :: res
    res%ch = x%ch // y%ch
  end function

  subroutine errors
    real :: a(10,10), b
    !ERROR: OPERATION= argument of REDUCE() must be a pure function of two data arguments
    b = reduce(a, f1)
    !ERROR: OPERATION= argument of REDUCE() must be a scalar function
    b = reduce(a, f2)
    !ERROR: OPERATION= argument of REDUCE() must be a pure function of two data arguments
    b = reduce(a, f3)
    !ERROR: OPERATION= argument of REDUCE() may not have dummy procedure arguments
    b = reduce(a, f4)
    !ERROR: OPERATION= argument of REDUCE() must have the same type as ARRAY=
    b = reduce(a, f5)
    !ERROR: Arguments of OPERATION= procedure of REDUCE() must be both scalar of the same type as ARRAY=, and neither allocatable, pointer, polymorphic, or optional
    b = reduce(a, f6)
    !ERROR: Arguments of OPERATION= procedure of REDUCE() must be both scalar of the same type as ARRAY=, and neither allocatable, pointer, polymorphic, or optional
    b = reduce(a, f7)
    !ERROR: Arguments of OPERATION= procedure of REDUCE() must be both scalar of the same type as ARRAY=, and neither allocatable, pointer, polymorphic, or optional
    b = reduce(a, f8)
    !ERROR: Arguments of OPERATION= procedure of REDUCE() must be both scalar of the same type as ARRAY=, and neither allocatable, pointer, polymorphic, or optional
    b = reduce(a, f9)
    !ERROR: If either argument of the OPERATION= procedure of REDUCE() has the ASYNCHRONOUS, VOLATILE, or TARGET attribute, both must have that attribute
    b = reduce(a, f10)
  end subroutine
  subroutine not_errors
    type(pdt(10)) :: a(10), b
    b = reduce(a, f11) ! check no bogus type incompatibility diagnostic
  end subroutine
end module
