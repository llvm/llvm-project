! RUN: %S/test_errors.sh %s %t %f18
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
