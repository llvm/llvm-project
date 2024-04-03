! RUN: %flang_fc1 -fdebug-unparse %s 2>&1

! Test that compiler directives can appear in various places.

#define PROC(KIND) \
  interface; integer(KIND) function foo(a); \
    integer(KIND), intent(in) :: a; \
    !dir$ ignore_tkr a; \
  end; end interface

!dir$ integer
module m
  !dir$ integer
  use iso_fortran_env
  !dir$ integer
  implicit integer(a-z)
  !dir$ integer
  !dir$ integer=64
  !dir$ integer = 64
  !dir$  integer = 64
  PROC(4)
  !dir$ optimize:1
  !dir$ optimize : 1
  !dir$ loop count (10000)
  !dir$ loop count (1, 500, 5000, 10000)
end
