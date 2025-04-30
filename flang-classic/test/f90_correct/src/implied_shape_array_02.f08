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
  !{error "PGF90-S-0155-Implied-shape array must be initialized with an array of the same rank - a1"}
  integer, parameter :: a1(*, *, 3) = a
  !{error "PGF90-S-0155-Implied-shape array must be initialized with an array of the same rank - a2"}
  integer, parameter :: a2(*) = reshape((/1, 10/), (/1, 1/))
  !{error "PGF90-S-0048-Illegal use of '*' in declaration of array"}
  !{error "PGF90-S-0048-Illegal use of '*' in declaration of array"}
  !{error "PGF90-S-0155-Implied-shape array must have the PARAMETER attribute - a3"}
  integer :: a3(*, *, *)
  !{error "PGF90-S-0048-Illegal use of '*' in declaration of array"}
  !{error "PGF90-S-0155-Implied-shape array must have the PARAMETER attribute - a4"}
  integer :: a4(*, *) = a
  !{error "PGF90-S-0143-a5 requires initializer"}
  integer, parameter :: a5(*)
  !{error "PGF90-S-0155-Implied-shape array must be initialized with a constant array - a6"}
  integer, parameter :: a6(*) = 0
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a7"}
  integer, parameter :: a7(l:*) = (/1, 10/)
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a8"}
  integer, parameter :: a8(2*l:*) = (/1, 10/)
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a9"}
  integer, parameter :: a9(-l:*) = (/1, 10/)
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a10"}
  integer, parameter :: a10(p%m:*) = (/1, 10/)
  !{error "PGF90-S-0155-Implied-shape array lower bound is not constant - a11"}
  integer, parameter :: a11(func(l):*) = (/1, 10/)
end program test
