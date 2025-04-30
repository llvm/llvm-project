!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM
! Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for USE statement when the local-name in subprogram has the same name
! with a unaccessible entity in the module which appears in the USE statement
! in host program.

module m1
  integer :: var1 = 1
end module

module m2
  use m1, var2 => var1
  private var2
  integer :: var3 = 2
end module

module m3
  integer :: var4 = 3
end module

program p
  use m2
  implicit none
  call test1()
  call test2()
  print *, "PASS"
contains
  subroutine test1()
    use m2, var2 => var3
    if (var2 /= 2) STOP 1
  end subroutine

  subroutine test2()
    use m3, var2 => var4
    if (var2 /= 3) STOP 2
  end subroutine
end program
