!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! 2d int array, maxloc with dim, mask, and back

program main
  integer :: res(4)
  integer, dimension(4,4) :: array
  integer :: expect(4)
  array = reshape((/4,2,9,-7,9,1,5,5,8,-1,-1,5,-7,5,9,-7/),shape(array))
  res = maxloc(array, dim = 1, mask = array .lt. 7, back=.true.)
  expect = (/1,4,4,2/)
  !print *, "base=true:", res
  call check(res, expect, 4)
  res = maxloc(array, dim = 1, mask = array .lt. 7, back=.false.)
  expect = (/1,3,4,2/)
  !print *, "back=false:", res
  call check(res, expect, 4)
end
