! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

MODULE test_c_ptr
IMPLICIT NONE

type :: handle_type
  integer :: iptr
end type

type c_ptr
  integer :: cptr
end type

CONTAINS

FUNCTION getptr(a)
class(handle_type),INTENT(in) :: a
TYPE(c_ptr) :: getptr
integer :: rslts, expect
expect = 4
rslts = a%iptr
call check(rslts, expect, 1)
getptr%cptr = 0

END FUNCTION getptr

END MODULE test_c_ptr

PROGRAM main
USE test_c_ptr
IMPLICIT NONE

TYPE(c_ptr) :: dummy
type(handle_type) :: input
input%iptr = 4
dummy = getptr(input)

END PROGRAM main
