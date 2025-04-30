!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for restrictions about LBOUND and UBOUND.

program test
  implicit none
  integer, allocatable :: x(:, :, :)

  allocate(x(2, 3, 4))
contains
  subroutine test_assumed_size(a)
    integer :: a(4:7, 9:*)
    !{error "PGF90-S-0084-Illegal use of symbol a - ubound of assumed size array is unknown"}
    print *, ubound(a)
    !{error "PGF90-S-0084-Illegal use of symbol a - ubound of assumed size array is unknown"}
    print *, ubound(a, 2)
  end subroutine
end program

