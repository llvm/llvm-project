! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensure that accessibility works on GENERIC statement
module m
  generic, public :: public => specific
  generic, private :: private => specific
 contains
  subroutine specific
  end
end
program main
  use m
  generic :: public => internal
  generic :: private => internal
  call public
  call public(1)
  !ERROR: No specific subroutine of generic 'private' matches the actual arguments
  call private
  call private(1)
 contains
  subroutine internal(n)
  end
end
