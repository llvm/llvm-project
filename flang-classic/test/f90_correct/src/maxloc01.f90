!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! 1d int array, maxloc with back

program main
  integer :: res(1)
  integer :: expect(2) = (/3, 2/)
  res = maxloc((/1,4,4,1/),back=.true.)
  !print *, "[1,4,4,1] back=true:", res
  call check(res, expect(1), 1)
  res = maxloc((/1,4,4,1/),back=.false.)
  !print *, "[1,4,4,1] back=false:", res
  call check(res, expect(2), 1)
end
