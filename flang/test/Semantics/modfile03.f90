! RUN: %python %S/test_modfile.py %s %flang_fc1
! Check modfile generation with use-association.

module m1
  integer :: x1
  integer, private :: x2
end
!Expect: m1.mod
!module m1
!integer(4)::x1
!integer(4),private::x2
!end

module m2
  use m1
  integer :: y1
end
!Expect: m2.mod
!module m2
!use m1,only:x1
!integer(4)::y1
!end

module m3
  use m2, z1 => x1
end
!Expect: m3.mod
!module m3
!use m2,only:y1
!use m2,only:z1=>x1
!end

module m4
  use m1
  use m2
end
!Expect: m4.mod
!module m4
!use m1,only:x1
!use m2,only:y1
!end

module m5a
  integer, parameter :: k1 = 4
  integer :: l1 = 2
  type t1
    real :: a
  end type
contains
  pure integer function f1(i)
    value :: i
    f1 = i
  end
end
!Expect: m5a.mod
!module m5a
! integer(4),parameter::k1=4_4
! integer(4)::l1
! type::t1
!  real(4)::a
! end type
!contains
! pure function f1(i)
!  integer(4),value::i
!  integer(4)::f1
! end
!end

module m5b
  use m5a, only: k2 => k1, l2 => l1, f2 => f1
  interface
    subroutine s(x, y)
      import f2, l2
      character(l2, k2) :: x
      character(f2(l2)) :: y
    end subroutine
  end interface
end
!Expect: m5b.mod
!module m5b
! use m5a,only:k2=>k1
! use m5a,only:l2=>l1
! use m5a,only:f2=>f1
! interface
!  subroutine s(x,y)
!   import::f2
!   import::l2
!   character(l2,4)::x
!   character(f2(l2),1)::y
!  end
! end interface
!end

module m6a
  type t1
  end type
end
!Expect: m6a.mod
!module m6a
! type::t1
! end type
!end

module m6b
  use m6a, only: t2 => t1
contains
  subroutine s(x)
    type(t2) :: x
  end
end
!Expect: m6b.mod
!module m6b
! use m6a,only:t2=>t1
!contains
! subroutine s(x)
!  type(t2)::x
! end
!end

module m6c
  use m6a, only: t2 => t1
  type, extends(t2) :: t
  end type
end
!Expect: m6c.mod
!module m6c
! use m6a,only:t2=>t1
! type,extends(t2)::t
! end type
!end

module m6d
  use m6a, only: t2 => t1
  type(t2), parameter :: p = t2()
end
!Expect: m6d.mod
!module m6d
! use m6a,only:t2=>t1
! type(t2),parameter::p=t2()
!end

module m6e
  use m6a, only: t2 => t1
  interface
    subroutine s(x)
      import t2
      type(t2) :: x
    end subroutine
  end interface
end
!Expect: m6e.mod
!module m6e
! use m6a,only:t2=>t1
! interface
!  subroutine s(x)
!   import::t2
!   type(t2)::x
!  end
! end interface
!end

module m7a
  real :: x
end
!Expect: m7a.mod
!module m7a
! real(4)::x
!end

module m7b
  use m7a
  private
end
!Expect: m7b.mod
!module m7b
! use m7a,only:x
! private::x
!end

module m8a
  private foo
  type t
   contains
    procedure, nopass :: foo
  end type
 contains
  pure integer function foo(n)
    integer, intent(in) :: n
    foo = n
  end
end
!Expect: m8a.mod
!module m8a
!type::t
!contains
!procedure,nopass::foo
!end type
!private::foo
!contains
!pure function foo(n)
!integer(4),intent(in)::n
!integer(4)::foo
!end
!end

module m8b
  use m8a
 contains
  subroutine foo(x,a)
    type(t), intent(in) :: x
    real a(x%foo(10))
  end
end
!Expect: m8b.mod
!module m8b
!use m8a,only:m8a$foo=>foo
!use m8a,only:t
!private::m8a$foo
!contains
!subroutine foo(x,a)
!type(t),intent(in)::x
!real(4)::a(1_8:int(m8a$foo(10_4),kind=8))
!end
!end

module m9a
  private
  public t
  type t
    integer n
   contains
    procedure f
  end type
 contains
  pure integer function f(x, k)
    class(t), intent(in) :: x
    integer, intent(in) :: k
    f = x%n + k
  end
end
!Expect: m9a.mod
!module m9a
!type::t
!integer(4)::n
!contains
!procedure::f
!end type
!private::f
!contains
!pure function f(x,k)
!class(t),intent(in)::x
!integer(4),intent(in)::k
!integer(4)::f
!end
!end

module m9b
  use m9a
 contains
  subroutine s(x, y)
    class(t), intent(in) :: x
    real y(x%f(x%n))
  end
end
!Expect: m9b.mod
!module m9b
!use m9a,only:t
!contains
!subroutine s(x,y)
!class(t),intent(in)::x
!real(4)::y(1_8:int(x%f(x%n),kind=8))
!end
!end
