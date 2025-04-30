!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! C1413 (R1416) If a submodule-name appears in the end-submodule-stmt, it shall
! be identical to the one in the submodule-stmt.
module ancestor
  interface 
    module subroutine hello
    end subroutine
  end interface
end module ancestor

! C1413 - negative - submodule-name in the end-submodule-stmt does not match
!  the name in the submodule-stmt. (compilation failure must report this)
submodule (ancestor) descendant
  contains
  module procedure hello
    print *, "hello world"
  end procedure
end submodule ancestor !{error "PGF90-S-0309-Incorrect name, ancestor, specified in END statement"}


program main
  use ancestor
  call hello
end program main


