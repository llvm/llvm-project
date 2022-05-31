! RUN: %python %S/test_errors.py %s %flang_fc1

module m
interface
  module subroutine dump()
  end subroutine
end interface
  integer, bind(c, name="a") :: x1
  integer, bind(c) :: x2
end

subroutine sub()
  !ERROR: A variable with BIND(C) attribute may only appear in the specification part of a module
  integer, bind(c, name="b") :: x3
  !ERROR: A variable with BIND(C) attribute may only appear in the specification part of a module
  integer, bind(c) :: x4
end

program main
  !ERROR: A variable with BIND(C) attribute may only appear in the specification part of a module
  integer, bind(c, name="c") :: x5
  !ERROR: A variable with BIND(C) attribute may only appear in the specification part of a module
  integer, bind(c) :: x6
end

submodule(m) m2
  !ERROR: A variable with BIND(C) attribute may only appear in the specification part of a module
  integer, bind(c, name="d") :: x7
  !ERROR: A variable with BIND(C) attribute may only appear in the specification part of a module
  integer, bind(c) :: x8
contains
  module procedure dump
  end procedure
end
