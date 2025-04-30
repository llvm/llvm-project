!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! 2d int array, maxloc with back

program main
  integer :: res(2)
  integer, dimension(2,2) :: array
  integer :: expect(2)
  array = reshape((/1,2,2,1/),shape(array))
  res = maxloc(array,back=.true.)
  expect = (/1,2/)
  !print *, "base=true:", res
  call check(res, expect, 2)
  res = maxloc(array,back=.false.)
  expect = (/2,1/)
  !print *, "back=false:", res
  call check(res, expect, 2)
end
