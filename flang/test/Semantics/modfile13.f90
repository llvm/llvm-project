! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
  character(2) :: z
  character(len=3) :: y
  character*4 :: x
  character :: w
  character(len=:), allocatable :: v
contains
  subroutine s(n, a, b, c, d)
    integer :: n
    character(len=n+1,kind=1) :: a
    character(n+2,2) :: b
    character*(n+3) :: c
    character(*) :: d
  end
end

!Expect: m.mod
!module m
!  character(2_4,1)::z
!  character(3_4,1)::y
!  character(4_8,1)::x
!  character(1_8,1)::w
!  character(:,1),allocatable::v
!contains
!  subroutine s(n,a,b,c,d)
!    integer(4)::n
!    character(n+1_4,1)::a
!    character(n+2_4,2)::b
!    character(n+3_4,1)::c
!    character(*,1)::d
!  end
!end
