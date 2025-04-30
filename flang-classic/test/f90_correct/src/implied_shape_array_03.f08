! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for implied-shape array

integer function func(arg)
  integer, intent(in) :: arg
  func = 2*arg
end function

program test
  implicit none
  type t
    integer :: m
  end type
  type(t) p
  integer :: l
  integer :: func
  integer, parameter :: a(3, 3) = reshape((/1, 10, 100, 2, 20, 200, 3, 30, 300/),&
                                          (/3, 3/))
  !{error "PGF90-S-0155-Implied-shape array must be initialized with a constant array - a1"}
  integer, parameter :: a1(*, *) = 0
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a2"}
  integer, parameter :: a2(l:*, *) = reshape((/1, 10, 100, 2, 20, 200, 3, 30, 300/),&
                                             (/3, 3/))
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a3"}
  integer, parameter :: a3(l:*, *) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a4"}
  integer, parameter :: a4(3*l:*, *) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a5"}
  integer, parameter :: a5(-l:*, *) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a6"}
  integer, parameter :: a6(p%m:*, *) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a7"}
  integer, parameter :: a7(func(l):*, *) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a8"}
  integer, parameter :: a8(l:*, l:*) = reshape((/1, 10, 100, 2, 20, 200, 3, 30, 300/),&
                                               (/3, 3/))
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a9"}
  integer, parameter :: a9(l:*, l:*) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a10"}
  integer, parameter :: a10(3*l:*, 3*l:*) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a11"}
  integer, parameter :: a11(-l:*, -l:*) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a12"}
  integer, parameter :: a12(p%m:*, p%m:*) = a
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a13"}
  integer, parameter :: a13(func(l):*, func(l):*) = a
  !{error "PGF90-S-0155-Implied-shape array must be initialized with a constant array - a14"}
  integer, parameter :: a14(*, *, *) = 0
  !{error "PGF90-S-0155-Implied-shape array must be initialized with an array of the same rank - a15"}
  integer, parameter :: a15(*, *, *) = a
  !{error "PGF90-S-0155-Implied-shape array must be initialized with an array of the same rank - a16"}
  integer, parameter :: a16(*, *, *) = reshape((/1, 10, 100, 2, 20, 200, 3, 30, 300/),&
                                               (/3, 3/))
end program test
