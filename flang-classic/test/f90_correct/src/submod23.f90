!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! C1412 (R1418) The ancestor-module-name shall be the name of a nonintrinsic
! module that declares a separate module procedure; the parent-submodule-name
! shall be the name of a descendant of that module.

!C1412 ancestor-module is nonintrinsic, declares a module procedure
module ancestor
  interface 
    module subroutine hello
    end subroutine
    module subroutine hello2
    end subroutine
  end interface
end module ancestor

module ancestor2
  interface 
    module subroutine hello
    end subroutine
    module subroutine hello2
    end subroutine
  end interface
end module ancestor2

submodule (ancestor) descendant23a
  contains
  module procedure hello
    print *, "hello world"
  end procedure
end submodule descendant23a

submodule (ancestor2) descendant23b
  contains
  module procedure hello
    print *, "hello world"
  end procedure
end submodule descendant23b

! C1412 - negative - parent-submodule-name (descendant23b) is not a descendant of ancestor
submodule (ancestor:descendant23b) grand_descendant !{error "PGF90-F-0004-Unable to open MODULE file ancestor-descendant23b.mod"}
  contains
  module procedure hello2
    print *, "hello again, world"
  end procedure
end submodule 

program main
  use ancestor
  call hello
  call hello2
end program main
