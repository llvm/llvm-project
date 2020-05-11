! RUN: %S/test_errors.sh %s %t %f18
! Forward references to derived types (error cases)
! C732 A parent-type-name shall be the name of a previously defined
! extensible type (7.5.7).

!ERROR: The derived type 'undef' was forward-referenced but not defined
type(undef) function f1()
  call sub(f1)
end function

!ERROR: The derived type 'undef' was forward-referenced but not defined
type(undef) function f2() result(r)
  call sub(r)
end function

!ERROR: The derived type 'undefpdt' was forward-referenced but not defined
type(undefpdt(1)) function f3()
  call sub(f3)
end function

!ERROR: The derived type 'undefpdt' was forward-referenced but not defined
type(undefpdt(1)) function f4() result(r)
  call sub(f4)
end function

!ERROR: 'bad' is not the name of a parameter for derived type 'pdt'
type(pdt(bad=1)) function f5()
  type :: pdt(good)
    integer, kind :: good = kind(0)
    integer(kind=good) :: n
  end type
end function

subroutine s1(q1)
  !ERROR: The derived type 'undef' was forward-referenced but not defined
  implicit type(undef)(q)
end subroutine

subroutine s2(q1)
  !ERROR: The derived type 'undefpdt' was forward-referenced but not defined
  implicit type(undefpdt(1))(q)
end subroutine

subroutine s3
  type :: t1
    !ERROR: Derived type 'undef' not found
    type(undef) :: x
  end type
end subroutine

subroutine s4
  type :: t1
    !ERROR: Derived type 'undefpdt' not found
    type(undefpdt(1)) :: x
  end type
end subroutine

subroutine s5(x)
  !ERROR: Derived type 'undef' not found
  type(undef) :: x
end subroutine

subroutine s6(x)
  !ERROR: Derived type 'undefpdt' not found
  type(undefpdt(1)) :: x
end subroutine

subroutine s7(x)
  !ERROR: Derived type 'undef' not found
  type, extends(undef) :: t
  end type
end subroutine
