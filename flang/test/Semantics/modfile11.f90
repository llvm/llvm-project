! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
  type t1(a, b, c)
    integer, kind :: a
    integer(8), len :: b, c
    integer :: d
  end type
  type, extends(t1) :: t2(e)
    integer, len :: e
  end type
  type, extends(t2), bind(c) :: t3
  end type
end

!Expect: m.mod
!module m
!  type::t1(a,b,c)
!    integer(4),kind::a
!    integer(8),len::b
!    integer(8),len::c
!    integer(4)::d
!  end type
!  type,extends(t1)::t2(e)
!    integer(4),len::e
!  end type
!  type,bind(c),extends(t2)::t3
!  end type
!end
