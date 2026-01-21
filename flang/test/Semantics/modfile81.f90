! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
  type dt
    procedure(sub), pointer, nopass :: p1 => sub
    procedure(sub), pointer, nopass :: p2 => null()
    procedure(sub), pointer, nopass :: p3
  end type
  procedure(sub), pointer :: p4 => sub
  procedure(sub), pointer :: p5 => null()
 contains
  subroutine sub
  end
end

!Expect: m.mod
!module m
!type::dt
!procedure(sub),nopass,pointer::p1=>sub
!procedure(sub),nopass,pointer::p2=>NULL()
!procedure(sub),nopass,pointer::p3
!end type
!intrinsic::null
!procedure(sub),pointer::p4
!procedure(sub),pointer::p5
!contains
!subroutine sub()
!end
!end
