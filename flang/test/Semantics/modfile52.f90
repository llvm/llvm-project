! RUN: %python %S/test_modfile.py %s %flang_fc1
! Ensure that procedure name or derived type name that has been shadowed
! behind a generic interface gets its proper USE statement in a module file.
module m1
 contains
  subroutine foo
  end subroutine
end module
module m2
  use m1
  interface foo
    procedure foo
  end interface
end module
module m3
  type foo
  end type
end module
module m4
  use m4
  interface foo
    procedure bar
  end interface
 contains
  integer function bar
  end function
end module

!Expect: m1.mod
!module m1
!contains
!subroutine foo()
!end
!end

!Expect: m2.mod
!module m2
!use m1,only:foo
!interface foo
!procedure::foo
!end interface
!end

!Expect: m3.mod
!module m3
!type::foo
!end type
!end

!Expect: m4.mod
!module m4
!interface foo
!procedure::bar
!end interface
!contains
!function bar()
!integer(4)::bar
!end
!end
