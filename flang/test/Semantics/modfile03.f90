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
