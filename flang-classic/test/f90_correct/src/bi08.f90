! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module mMmM

common n

contains

  subroutine mM1() ! mangled linker name
    n = n + 1
  end subroutine

  subroutine mM2() bind(C)
    n = n + 1
  end subroutine

  subroutine mM3() bind(C,name="") ! mangled linker name
    n = n + 1
  end subroutine

  subroutine mM4() bind(C,name="mM4")
    n = n + 1
  end subroutine

end module mMmM

subroutine sS1() ! mangled linker name
  common n
  n = n + 1
end subroutine

subroutine sS2() bind(C)
  common n
  n = n + 1
end subroutine

subroutine sS3() bind(C,name="") ! mangled linker name
  common n
  n = n + 1
end subroutine

subroutine sS4() bind(C,name="sS4")
  common n
  n = n + 1
end subroutine

! --------------------

use mMmM

interface
  subroutine sS2() bind(C)
  end subroutine
  subroutine sS3() bind(C,name="")
  end subroutine
  subroutine sS4() bind(C,name="sS4")
  end subroutine
  subroutine cC() bind(C)
  end subroutine
end interface

n = 0

call mM1
call mM2
call mM3
call mM4

call sS1
call sS2
call sS3
call sS4

call cC

if (n .eq. 12) print*, 'PASS'
if (n .ne. 12) print*, 'FAIL: expected 12 calls; made', n

end
