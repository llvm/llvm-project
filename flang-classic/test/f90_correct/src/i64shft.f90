! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program i64shft
  integer(8) :: res(4), expect(4), x, y

  x = 1
  y = 32
  res = do_shfts(x, y)

  expect = 4294967296

  call check(res, expect, 4)

  contains

   function do_shfts(x, y) result(r)
     integer(8) :: x, y, r(4)
     integer(8) :: z = ishft(1, 32) ! compile-time

     r(1) = z
     r(2) = ishft(z, 1)
     r(2) = ishft(r(2), -1)
     r(3) = ishft(x,32)
     r(4) = ishft(x,y)
   end function
end program
