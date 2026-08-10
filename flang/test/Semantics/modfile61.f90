! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
  type t1
    procedure(p1), pointer, nopass :: p
  end type
  type t2
    procedure(p2), pointer, nopass :: p
  end type
  type t3
    procedure(p4), pointer, nopass :: p
  end type
  type t4
    procedure(p6), pointer, nopass :: p
  end type
  type t5
    procedure(p7), pointer, nopass :: p
  end type
  interface
    subroutine p1
    end
    subroutine p2
    end
    subroutine p3
    end
    subroutine p4
    end
    subroutine p5(c)
      import
      type(t3), intent(in) :: c
    end
    subroutine p6(d)
      import
      type(t5), intent(in) :: d
    end
    subroutine p7
    end
    subroutine p8
    end
    function f(a,b,dp)
      import
      type(t1), intent(in) :: a
      type, extends(t2) :: localt1
        procedure(p3), pointer, nopass :: p
      end type
      type, extends(localt1) :: localt2
       contains
        procedure, nopass :: p8
      end type
      type(localt2), intent(in) :: b
      procedure(p5) dp
      type(t4), pointer :: f
    end
  end interface
end

!Expect: m.mod
!module m
!type::t1
!procedure(p1),nopass,pointer::p
!end type
!type::t2
!procedure(p2),nopass,pointer::p
!end type
!type::t3
!procedure(p4),nopass,pointer::p
!end type
!type::t4
!procedure(p6),nopass,pointer::p
!end type
!type::t5
!procedure(p7),nopass,pointer::p
!end type
!interface
!subroutine p1()
!end
!end interface
!interface
!subroutine p2()
!end
!end interface
!interface
!subroutine p3()
!end
!end interface
!interface
!subroutine p4()
!end
!end interface
!interface
!subroutine p5(c)
!import::t3
!type(t3),intent(in)::c
!end
!end interface
!interface
!subroutine p6(d)
!import::t5
!type(t5),intent(in)::d
!end
!end interface
!interface
!subroutine p7()
!end
!end interface
!interface
!subroutine p8()
!end
!end interface
!interface
!function f(a,b,dp)
!import::p3
!import::p5
!import::p8
!import::t1
!import::t2
!import::t4
!type(t1),intent(in)::a
!type,extends(t2)::localt1
!procedure(p3),nopass,pointer::p
!end type
!type,extends(localt1)::localt2
!contains
!procedure,nopass::p8
!end type
!type(localt2),intent(in)::b
!procedure(p5)::dp
!type(t4),pointer::f
!end
!end interface
!end
