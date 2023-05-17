! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
module m
 contains
  subroutine s1(x)
    character(3) :: x
  end
  subroutine s2(x)
    character(3) :: x(1)
  end
  subroutine s3(x)
    character(3) :: x(:)
  end
  subroutine s4(x)
    character(3) :: x(..)
  end
  subroutine s5(x)
    character(3), allocatable :: x
  end
  subroutine s6(x)
    character(3), pointer :: x
  end
end

program test
  use m
  character(2) short, shortarr(1)
  character(2), allocatable :: shortalloc
  character(2), pointer :: shortptr
  character(4) long, longarr(1)
  character(4), allocatable :: longalloc
  character(4), pointer :: longptr
  !WARNING: Actual argument variable length '2' is less than expected length '3'
  call s1(short)
  !WARNING: Actual argument variable length '2' is less than expected length '3'
  call s2(shortarr)
  !ERROR: Actual argument variable length '2' does not match the expected length '3'
  call s3(shortarr)
  !ERROR: Actual argument variable length '2' does not match the expected length '3'
  call s4(shortarr)
  !ERROR: Actual argument variable length '2' does not match the expected length '3'
  call s5(shortalloc)
  !ERROR: Actual argument variable length '2' does not match the expected length '3'
  !ERROR: Target type CHARACTER(KIND=1,LEN=2_8) is not compatible with pointer type CHARACTER(KIND=1,LEN=3_8)
  call s6(shortptr)
  call s1(long) ! ok
  call s2(longarr) ! ok
  !ERROR: Actual argument variable length '4' does not match the expected length '3'
  call s3(longarr)
  !ERROR: Actual argument variable length '4' does not match the expected length '3'
  call s4(longarr)
  !ERROR: Actual argument variable length '4' does not match the expected length '3'
  call s5(longalloc)
  !ERROR: Actual argument variable length '4' does not match the expected length '3'
  !ERROR: Target type CHARACTER(KIND=1,LEN=4_8) is not compatible with pointer type CHARACTER(KIND=1,LEN=3_8)
  call s6(longptr)
end
