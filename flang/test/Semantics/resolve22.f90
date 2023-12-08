! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s1
  !OK: interface followed by type with same name
  interface t
  end interface
  type t
  end type
  type(t) :: x
  x = t()
end subroutine

subroutine s2
  !OK: type followed by interface with same name
  type t
  end type
  interface t
  end interface
  type(t) :: x
  x = t()
end subroutine

subroutine s3
  type t
  end type
  interface t
  end interface
  !ERROR: 't' is already declared in this scoping unit
  type t
  end type
  type(t) :: x
  x = t()
end subroutine

module m4
  type t1
    class(t2), pointer :: p => null()
  end type
  type t2
  end type
  interface t2
    procedure ctor
  end interface
 contains
  function ctor()
    type(t2) ctor
  end function
end module
