!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! maxloc with minimum value of integer type as elements

program t
  integer, dimension(2) :: rslt, expect
  integer, dimension(2, 2) :: m
  m = -huge(integer) - 1
  rslt = maxloc(m, dim = 1)
  expect = (/1, 1/)
  call check(rslt, expect, 2)
end
