! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test mod-file generation for F2023 explicit-shape bounds using rank-1
! integer arrays (ExplicitShapeBoundsSpec / RankOneBoundElement).

! PARAMETER rank-1 array as upper bounds
module m1
  integer, parameter :: dims(3) = [5, 10, 15]
  real :: a(dims)
end module

!Expect: m1.mod
!module m1
!integer(4),parameter::dims(1_8:3_8)=[INTEGER(4)::5_4,10_4,15_4]
!real(4)::a(1_8:[INTEGER(8)::5_8,10_8,15_8])
!end

! Rank-1 dummy as upper bounds
module m2
contains
subroutine sub1(n,a)
  integer, intent(in) :: n(3)
  real :: a(n)
end subroutine
end module

!Expect: m2.mod
!module m2
!contains
!subroutine sub1(n,a)
!integer(4),intent(in)::n(1_8:3_8)
!real(4)::a(1_8:__builtin_int(n,kind=8))
!end
!end

! Both lower and upper rank-1 bounds
module m3
contains
subroutine sub2(lb,ub,a)
  integer, intent(in) :: lb(2), ub(2)
  real :: a(lb:ub)
end subroutine
end module

!Expect: m3.mod
!module m3
!contains
!subroutine sub2(lb,ub,a)
!integer(4),intent(in)::lb(1_8:2_8)
!integer(4),intent(in)::ub(1_8:2_8)
!real(4)::a(__builtin_int(lb,kind=8):__builtin_int(ub,kind=8))
!end
!end
