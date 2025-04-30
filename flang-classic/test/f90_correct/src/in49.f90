! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! intrinsic statement in a contained routine

call ss

contains
  subroutine ss
    integer, intrinsic :: len
    call invoke(len, 'PASS')
  end

  subroutine invoke(f, string)
    integer f
    character*(*) string
    if (f(string) .eq. len(string)) then
      print*, string
      return
    endif
    print*, 'FAIL'
  end
end
