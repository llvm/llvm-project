!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM
! Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for USE statement when redeclare local-name or only-use-name identifier
! in subprogram.

module m
  integer :: var = 1
end module

program p
  implicit none
contains
  subroutine test1()
    use m, localvar => var
    !{error "PGF90-S-0155-var is use associated and cannot be redeclared"}
    integer :: localvar
  end subroutine

  subroutine test2()
    use m, only: var
    !{error "PGF90-S-0155-var is use associated and cannot be redeclared"}
    integer :: var
  end subroutine
end program
