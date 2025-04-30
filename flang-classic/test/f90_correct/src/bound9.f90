!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for restrictions about LBOUND and UBOUND.

program test
  implicit none
  integer :: x(2,3,4)
  type t
    integer :: y
  end type
  type(t) :: a = t(1)
  !{error "PGF90-S-0423-Constant DIM= argument is out of range"}
  print *, ubound(x, 4)
  !{error "PGF90-S-0423-Constant DIM= argument is out of range"}
  print *, lbound(x, 0)
  !{error "PGF90-S-0074-Illegal number or type of arguments to ubound - keyword argument *dim"}
  print *, ubound(x, a)
  !{error "PGF90-S-0074-Illegal number or type of arguments to lbound - keyword argument *kind"}
  print *, lbound(x, a%y, a)
contains
  subroutine test_assumed_rank(a)
    integer :: a(..)
    !{error "PGF90-S-0423-Constant DIM= argument is out of range"}
    print *, ubound(a, DIM=200)
  end subroutine
end program

