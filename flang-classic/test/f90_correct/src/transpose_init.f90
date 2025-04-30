!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

!* Test checking tranpose during initialization
program test
      integer, parameter :: num = 1
      integer rslts(num), expect(num)
      data expect / 1 /
      integer, parameter :: arr(2, 3) = RESHAPE((/1, 2, 3, 4, 5, 6/), (/2, 3/))
      integer, parameter :: exp_transpose_arr(3, 2) = RESHAPE((/1, 3, 5, 2, 4, 6/), (/3, 2/))
      integer :: transpose_arr(3, 2) = TRANSPOSE(arr)

      if (all(transpose_arr .eq. exp_transpose_arr)) then
          rslts(1) = 1
      else
          rslts(1) = 0
          print *, 'tranpose_arr vs exp_transpose_arr mismatch'
      endif

      call check(rslts, expect, num)
end program
