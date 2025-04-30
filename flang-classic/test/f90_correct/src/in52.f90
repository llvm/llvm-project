! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! intrinsic declarations in host and internal routine

program pp
  integer(kind=4), intrinsic :: len_trim
  call ss

contains
  integer function kk
    integer(kind=4) :: len_trim
    len_trim = 3
    kk = len_trim
  end

  subroutine ss
    integer(kind=4), intrinsic :: len_trim
    character*4 :: s = 'FAIL'
    if (len_trim(s) - kk() .eq. 1) then
      print*, 'PASS'
    else
      print*, s
    endif
  end
end
