! RUN: %python %S/test_errors.py %s %flang_fc1
! Legacy DATA-style initialization on declarations that are in error.
! Each shape here used to crash instead of emitting the error.
program main
  type foo
  end type foo
  !ERROR: 'foo' is already declared in this scoping unit
  integer foo(1) /2/
end program main
subroutine s1
  type foo
  end type foo
  !ERROR: 'foo' is already declared in this scoping unit
  integer foo /1/
end subroutine
subroutine s2
  interface foo
  end interface
  !ERROR: 'foo' is already declared in this scoping unit
  integer foo /1/
end subroutine
subroutine s3
  type foo
  end type foo
  !ERROR: 'foo' is already declared in this scoping unit
  character foo*3 /'abc'/
end subroutine
subroutine s4
  type foo
  end type foo
  !ERROR: 'foo' is already declared in this scoping unit
  integer foo(1) /2/, ok(1) /3/
end subroutine
subroutine s5
  integer :: x
  namelist /foo/ x
  !ERROR: 'foo' is already declared in this scoping unit
  integer foo /1/
end subroutine
