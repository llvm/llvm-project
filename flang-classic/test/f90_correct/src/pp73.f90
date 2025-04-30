! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! 
! Tests subscripted pointer expressions that point to non-contiguous array
! sections.

program main
  integer, pointer :: p1(:), p2(:)
  integer, target :: a(20)
  integer :: expect(8) = [3, 5, 7, 9, 11, 13, 15, 17]
  integer :: j
  a = [(j,j=1,20)]
  p1 => a(::2)
  p2(1:8) => p1(2:9)
  if (all(p2 .eq. expect)) then
    print *, 'PASS'
  else
    print *, 'FAIL'
  endif
end program main
