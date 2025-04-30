! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


! Test use-associated (same variable and function names)from two modules n and
! m.

module m
  integer, parameter :: i = -1
  interface
    module subroutine show_i
    end subroutine show_i
  end interface
contains
  integer function times_two (arg)
    integer :: arg
    times_two = -2*arg
  end function
end module m

module n
  integer, parameter :: i = 2
contains
  integer function times_two (arg)
    integer :: arg
    times_two = 2*arg
  end function
end module n

submodule (m) sm
  use n
contains
  module subroutine show_i
    if (i .ne. 2) then
      print *, "FAIL"
    else
      print *, "PASS"
    endif
    if (times_two (i) .ne. 4) then
      print *, "FAIL"
    else
      print *, "PASS"
    endif
  end subroutine show_i
end submodule sm

program p
  use m
  call show_i
  if (i .ne. -1) then
    print *, "FAIL"
  else
    print *, "PASS"
  endif
  if (times_two (i) .ne. 2) then
    print *, "FAIL"
  else
    print *, "PASS"
  endif
end program
