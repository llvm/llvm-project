! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!
! Tests F2003 allocatable semantics with CHAR intrinsic.
 
program p
  logical rslt(4), expect(4)  
  character(1), allocatable :: c1, c2
  integer :: i = 77 
  
  rslt = .false.
  expect = .true.
  c1 = CHAR(i)
  c2 = ACHAR(i)
  
  rslt(1) = allocated(c1)
  rslt(2) = allocated(c2)
  if (rslt(1)) then
    rslt(3) = c1 .eq. 'M'
  endif
  if (rslt(2)) then
    rslt(4) = c2 .eq. 'M'
  endif
  
  call check(rslt, expect, 4)
end program p
