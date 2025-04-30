!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! 2d int array, maxloc with dim, and back

program main
  integer :: res(4)
  integer, dimension(3,4) :: array
  integer :: expect(4)
  array = reshape((/4,3,-1,0,1,-4,5,-2,5,2,6,5/),shape(array))
  res = maxloc(array, dim = 1, back=.true.)
  expect = (/1,2,3,2/)
  call check(res, expect, 4)
  !print *, "base=true:", res
  res = maxloc(array, dim = 1, back=.false.)
  expect = (/1,2,1,2/)
  call check(res, expect, 4)
  !print *, "back=false:", res
end
