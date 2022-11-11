! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
  character(2), parameter :: prefix = 'c_'
  integer, bind(c, name='c_a') :: a
  procedure(sub), bind(c, name=prefix//'b') :: b
  type, bind(c) :: t
    real :: c
  end type
contains
  subroutine sub() bind(c, name='sub')
  end
end

!Expect: m.mod
!module m
!  character(2_4,1),parameter::prefix="c_"
!  integer(4),bind(c, name="c_a")::a
!  procedure(sub),bind(c, name="c_b")::b
!  type,bind(c)::t
!    real(4)::c
!  end type
!contains
!  subroutine sub() bind(c, name="sub")
!  end
!end
