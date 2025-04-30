!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM
! Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for USE statement when the local-name and use-name in rename of generic
! interface have the same name.

module m1
  interface gnr
    module procedure func
  end interface
contains
  integer function func(x)
    integer :: x
    func = x
  end function
end module

module m2
end module

program p
  implicit none
  call subr()
  print *, "PASS"
contains
  subroutine subr()
    use m1
    use m1, only: gnr => gnr
    use m2
    if (gnr(100) /= 100) STOP 1
  end subroutine
end program
