!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for implied-do containing function call

program p
  implicit none
  integer :: i, j
  integer :: cmp1(12) = [5, 10, 1, 5, 10, 2, 2, 5, 10, 3, 3, 3]
  integer :: cmp2(12) = [5, 10, 1, 1, 5, 10, 2, 2, 5, 10, 3, 3]
  integer :: cmp3(17) = [3, 4, 1, 3, 4, 2, 2, 3, 4, 2, 2, 3, 4, 4, 4, 4, 4]

  if (size(cmp1) /= size((/ ((/ 5, 10, gen (i) /), i = 1, 3) /))) &
    STOP 1
  if (any(shape(cmp1) /= shape((/ ((/ 5, 10, gen (i) /), i = 1, 3) /)))) &
    STOP 2
  if (any(lbound(cmp1) /= lbound((/ ((/ 5, 10, gen (i) /), i = 1, 3) /)))) &
    STOP 3
  if (any(ubound(cmp1) /= ubound((/ ((/ 5, 10, gen (i) /), i = 1, 3) /)))) &
    STOP 4
  if (any(cmp1 /= (/ ((/ 5, 10, gen (i) /), i = 1, 3) /))) &
    STOP 5

  if (size(cmp2) /= size((/ ((/ 5, 10, gen2 (i) /), i = 1, 3) /))) &
    STOP 6
  if (any(shape(cmp2) /= shape((/ ((/ 5, 10, gen2 (i) /), i = 1, 3) /)))) &
    STOP 7
  if (any(lbound(cmp2) /= lbound((/ ((/ 5, 10, gen2 (i) /), i = 1, 3) /)))) &
    STOP 8
  if (any(ubound(cmp2) /= ubound((/ ((/ 5, 10, gen2 (i) /), i = 1, 3) /)))) &
    STOP 9
  if (any(cmp2 /= (/ ((/ 5, 10, gen2 (i) /), i = 1, 3) /))) &
    STOP 10

  if (size(cmp3) /= size((/((3, 4, gen3(i, j), i = 1, 2), j = 1, 2)/))) &
    STOP 11
  if (any(shape(cmp3) /= shape((/((3, 4, gen3(i, j), i = 1, 2), j = 1, 2)/)))) &
    STOP 12
  if (any(lbound(cmp3) /= lbound((/((3, 4, gen3(i, j), i = 1, 2), j = 1, 2)/)))) &
    STOP 13
  if (any(ubound(cmp3) /= ubound((/((3, 4, gen3(i, j), i = 1, 2), j = 1, 2)/)))) &
    STOP 14
  if (any(cmp3 /= (/((3, 4, gen3(i, j), i = 1, 2), j = 1, 2)/))) &
    STOP 15

  print *, "PASS"
contains
  function gen (n) result(z)
    integer, dimension (:), pointer :: z
    integer :: n
    allocate (z (n))
    z = n
  end function gen

  function gen2 (n) result(z)
    integer, dimension (2) :: z
    integer :: n
    z = n
  end function gen2

  function gen3(n1, n2) result(z)
    integer, pointer :: z(:,:)
    integer :: n1, n2
    allocate(z(n1, n2))
    z = n1*n2
  end function gen3

end program
