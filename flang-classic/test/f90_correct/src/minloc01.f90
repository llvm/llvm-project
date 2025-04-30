!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! 1d int array, minloc with back

program main
  integer :: res(1)
  integer :: expect(2) = (/4,1/)
  res = minloc((/1,4,4,1/),back=.true.)
  call check(res, expect(1), 1)
  !print *, "[1,4,4,1] back=true:", res
  res = minloc((/1,4,4,1/),back=.false.)
  call check(res, expect(2), 1)
  !print *, "[1,4,4,1] back=false:", res
end
