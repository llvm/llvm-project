! RUN: %f18 -funparse %s 2>&1

! Test that compiler directives can appear in various places.

!dir$ integer
module m
  !dir$ integer
  use iso_fortran_env
  !dir$ integer
  implicit integer(a-z)
  !dir$ integer
end
