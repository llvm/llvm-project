!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
! 2d int array, minloc with back

program main
  integer :: res(2)
  integer, dimension(2,2) :: array
  integer :: expect(2)
  array = reshape((/1,2,2,1/),shape(array))
  res = minloc(array,back=.true.)
  expect = (/2,2/)
  call check(res, expect, 2)
  !print *, "base=true:", res
  res = minloc(array,back=.false.)
  expect = (/1,1/)
  call check(res, expect, 2)
  !print *, "back=false:", res
end
