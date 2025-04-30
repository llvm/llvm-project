!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! 1d char array, maxloc with back

program main
  integer :: res(1)
  integer :: expect(2) = (/3, 2/)
  res = maxloc((/"a","b","b","a"/),back=.true.)
  call check(res, expect(1), 1)
  !print *, "back=true:", res
  res = maxloc((/"a","b","b","a"/),back=.false.)
  call check(res, expect(2), 1)
  !print *, "back=false:", res
end
