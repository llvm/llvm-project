!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! 2d int array, maxloc with mask, and back

program main
  integer :: res(2)
  integer, dimension(3,4) :: array
  integer :: expect(2)
  array = reshape((/4,3,-1,0,1,-4,4,-2,5,2,6,5/),shape(array))
  res = maxloc(array, mask = array .lt. 5, back=.true.)
  expect = (/1,3/)
  call check(res, expect, 2)
  !print *, "base=true:", res
  res = maxloc(array, mask = array .lt. 5, back=.false.)
  expect = (/1,1/)
  call check(res, expect, 2)
  !print *, "back=false:", res
end
