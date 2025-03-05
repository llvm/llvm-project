! RUN: %python %S/test_modfile.py %s %flang_fc1
module mod0
  interface proc
    module procedure proc
  end interface
 contains
  subroutine proc
  end
end
module mod1
  use mod0,renamed_proc=>proc
  procedure(renamed_proc),pointer :: p
end module

!Expect: mod0.mod
!module mod0
!interface proc
!procedure::proc
!end interface
!contains
!subroutine proc()
!end
!end

!Expect: mod1.mod
!module mod1
!use mod0,only:renamed_proc=>proc
!procedure(renamed_proc),pointer::p
!end
