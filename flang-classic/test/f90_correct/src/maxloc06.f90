!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! 2d int array, maxloc with back

program main
  integer :: res(2)
  integer, dimension(4,4) :: array
  integer :: expect(2)
  array = reshape((/4,2,9,-7,9,1,5,5,8,-1,-1,5,-7,5,9,-7/),shape(array))
  res = maxloc(array,back=.true.)
  expect = (/3,4/)
  call check(res, expect, 2)
  !print *, "base=true:", res
  res = maxloc(array,back=.false.)
  expect = (/3,1/)
  call check(res, expect, 2)
  !print *, "back=false:", res
end
