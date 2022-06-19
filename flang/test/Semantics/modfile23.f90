! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test that subprogram interfaces get all of the symbols that they need.

module m1
  integer(8) :: i
  type t1
    sequence
    integer :: j
  end type
  type t2
  end type
end
!Expect: m1.mod
!module m1
! integer(8)::i
! type::t1
!  sequence
!  integer(4)::j
! end type
! type::t2
! end type
!end

module m2
  integer(8) :: k
contains
  subroutine s(a, j)
    use m1
    integer(8) :: j
    real :: a(i:j,1:k)  ! need i from m1
  end
end
!Expect: m2.mod
!module m2
! integer(8)::k
!contains
! subroutine s(a,j)
!  use m1,only:i
!  integer(8)::j
!  real(4)::a(i:j,1_8:k)
! end
!end

module m3
  implicit none
contains
  subroutine s(b, n)
    type t2
    end type
    type t4(l)
      integer, len :: l
      type(t2) :: x  ! need t2
    end type
    integer :: n
    type(t4(n)) :: b
  end
end module
!Expect: m3.mod
!module m3
!contains
! subroutine s(b,n)
!  integer(4)::n
!  type::t2
!  end type
!  type::t4(l)
!   integer(4),len::l
!   type(t2)::x
!  end type
!  type(t4(l=n))::b
! end
!end

module m4
contains
  subroutine s1(a)
    use m1
    common /c/x,n  ! x is needed
    integer(8) :: n
    real :: a(n)
    type(t1) :: x
  end
end
!Expect: m4.mod
!module m4
!contains
! subroutine s1(a)
!  use m1,only:t1
!  type(t1)::x
!  common/c/x,n
!  integer(8)::n
!  real(4)::a(1_8:n)
! end
!end

module m5
  type t5
  end type
  interface
    subroutine s(x1,x5)
      use m1
      import :: t5
      type(t1) :: x1
      type(t5) :: x5
    end subroutine
  end interface
end
!Expect: m5.mod
!module m5
! type::t5
! end type
! interface
!  subroutine s(x1,x5)
!   use m1,only:t1
!   import::t5
!   type(t1)::x1
!   type(t5)::x5
!  end
! end interface
!end

module m6
contains
  subroutine s(x)
    use m1
    type, extends(t2) :: t6
    end type
    type, extends(t6) :: t7
    end type
    type(t7) :: x
  end
end
!Expect: m6.mod
!module m6
!contains
! subroutine s(x)
!  use m1,only:t2
!  type,extends(t2)::t6
!  end type
!  type,extends(t6)::t7
!  end type
!  type(t7)::x
! end
!end

module m7
  type :: t5(l)
    integer, len :: l
  end type
contains
  subroutine s1(x)
    use m1
    type(t5(i)) :: x
  end subroutine
  subroutine s2(x)
    use m1
    character(i) :: x
  end subroutine
end
!Expect: m7.mod
!module m7
! type::t5(l)
!  integer(4),len::l
! end type
!contains
! subroutine s1(x)
!  use m1,only:i
!  type(t5(l=int(i,kind=4)))::x
! end
! subroutine s2(x)
!  use m1,only:i
!  character(i,1)::x
! end
!end

module m8
  use m1, only: t1, t2
  interface
    subroutine s1(x)
      import
      type(t1) :: x
    end subroutine
    subroutine s2(x)
      import :: t2
      type(t2) :: x
    end subroutine
  end interface
end
!Expect: m8.mod
!module m8
! use m1,only:t1
! use m1,only:t2
! interface
!  subroutine s1(x)
!   import::t1
!   type(t1)::x
!  end
! end interface
! interface
!  subroutine s2(x)
!   import::t2
!   type(t2)::x
!  end
! end interface
!end
