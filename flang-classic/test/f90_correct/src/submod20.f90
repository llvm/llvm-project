!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! C1413 (R1416) If a submodule-name appears in the end-submodule-stmt, it shall
! be identical to the one in the submodule-stmt.

! C1412 (R1418) The ancestor-module-name shall be the name of a nonintrinsic
! module that declares a separate module procedure; the parent-submodule-name
! shall be the name of a descendant of that module.

! C1411 (R1416) A submodule specification-part shall not contain a format-stmt


!C1412 ancestor-module is nonintrinsic, declares a module procedure
module ancestor
  interface 
    module subroutine hello
    end subroutine
    module subroutine hello2
    end subroutine
  end interface
end module ancestor

! C1411 - submodule specification-part does not contain a format-stmt
! C1413 - test that matching submodule-name is accepted
submodule (ancestor) descendant
  contains
  module procedure hello
!    print *, "hello world"
     write(*,"(a)",advance="no")" PA"  
  end procedure
end submodule descendant

! C1411 - submodule specification-part does not contain a format-stmt
! C1412 - parent-submodule-name (descendant) is a descendant of ancestor
! C1413 - test that end-submodule-stmt without submodule-name is accepted
submodule (ancestor:descendant) descendant2
  contains
  module procedure hello2
     write(*,"(a)",advance="no") "SS "
     print *, ""
!    print *, "hello again, world"
  end procedure
end submodule 


program main
  use ancestor
  call hello
  call hello2
end program main


