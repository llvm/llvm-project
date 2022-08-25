! RUN: %python %S/test_modfile.py %s %flang_fc1
module m1
  interface foo
    module procedure foo
  end interface
 contains
  subroutine foo
  end subroutine
end module
module m2
  use m1, bar => foo
  interface baz
    module procedure bar ! must not be replaced in module file with "foo"
  end interface
end module

!Expect: m1.mod
!module m1
!interface foo
!procedure::foo
!end interface
!contains
!subroutine foo()
!end
!end

!Expect: m2.mod
!module m2
!use m1,only:bar=>foo
!interface baz
!procedure::bar
!end interface
!end
