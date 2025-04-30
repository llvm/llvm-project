! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test which tracks allocatable behaviour

program test
      integer, parameter :: num = 1
      integer rslts(num), expect(num)
      data expect / 1 /
      integer, allocatable :: arr(:)
      integer :: exp_arr(5) = (/1,2,3,4,5/)
      arr = get_array()

      if (all(arr .eq. exp_arr)) then
          rslts(1) = 1
      else
          rslts(1) = 0
          write (*,*), arr, '-expected ', exp_arr
      endif

      call check(rslts, expect, num)

contains

      function get_array() result(larr)
          integer :: larr(5)
          larr = (/1,2,3,4,5/)
      end function
end program
