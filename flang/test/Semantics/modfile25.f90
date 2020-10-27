! RUN: %S/test_modfile.sh %s %t %f18
! Test compile-time analysis of shapes.

module m1
  integer(8), parameter :: a0s(*) = shape(3.14159)
  real :: a1(5,5,5)
  integer(8), parameter :: a1s(*) = shape(a1)
  integer(8), parameter :: a1ss(*) = shape(a1s)
  integer(8), parameter :: a1sss(*) = shape(a1ss)
  integer(8), parameter :: a1rs(*) = [rank(a1),rank(a1s),rank(a1ss),rank(a1sss)]
  integer(8), parameter :: a1n(*) = [size(a1),size(a1,1),size(a1,2)]
  integer(8), parameter :: a1sn(*) = [size(a1s),size(a1ss),size(a1sss)]
  integer(8), parameter :: ac1s(*) = shape([1])
  integer(8), parameter :: ac2s(*) = shape([1,2,3])
  integer(8), parameter :: ac3s(*) = shape([(1,j=1,4)])
  integer(8), parameter :: ac3bs(*) = shape([(1,j=4,1,-1)])
  integer(8), parameter :: ac4s(*) = shape([((j,k,j*k,k=1,3),j=1,4)])
  integer(8), parameter :: ac5s(*) = shape([((0,k=5,1,-2),j=9,2,-3)])
  integer(8), parameter :: rss(*) = shape(reshape([(0,j=1,90)], -[2,3]*(-[5_8,3_8])))
 contains
  subroutine subr(x,n1,n2)
    real, intent(in) :: x(:,:)
    integer, intent(in) :: n1(3), n2(:)
    real, allocatable :: a(:,:,:)
    ! the following fail if we don't handle empty strings
    Character(0) :: ch1(1,2,3) = Reshape([('',n=1,1*2*3)],[1,2,3])
    Character(0) :: ch2(3) = reshape(['','',''], [3])
    a = reshape(x,n1)
    a = reshape(x,n2(10:30:9)) ! fails if we can't figure out triplet shape
  end subroutine
end module m1
!Expect: m1.mod
! module m1
! integer(8),parameter::a0s(1_8:*)=[INTEGER(8)::]
! intrinsic::shape
! real(4)::a1(1_8:5_8,1_8:5_8,1_8:5_8)
! integer(8),parameter::a1s(1_8:*)=[INTEGER(8)::5_8,5_8,5_8]
! integer(8),parameter::a1ss(1_8:*)=[INTEGER(8)::3_8]
! integer(8),parameter::a1sss(1_8:*)=[INTEGER(8)::1_8]
! integer(8),parameter::a1rs(1_8:*)=[INTEGER(8)::3_8,1_8,1_8,1_8]
! integer(8),parameter::a1n(1_8:*)=[INTEGER(8)::125_8,5_8,5_8]
! integer(8),parameter::a1sn(1_8:*)=[INTEGER(8)::3_8,1_8,1_8]
! integer(8),parameter::ac1s(1_8:*)=[INTEGER(8)::1_8]
! integer(8),parameter::ac2s(1_8:*)=[INTEGER(8)::3_8]
! integer(8),parameter::ac3s(1_8:*)=[INTEGER(8)::4_8]
! integer(8),parameter::ac3bs(1_8:*)=[INTEGER(8)::4_8]
! integer(8),parameter::ac4s(1_8:*)=[INTEGER(8)::36_8]
! integer(8),parameter::ac5s(1_8:*)=[INTEGER(8)::9_8]
! integer(8),parameter::rss(1_8:*)=[INTEGER(8)::10_8,9_8]
! intrinsic::reshape
! contains
! subroutine subr(x,n1,n2)
! real(4),intent(in)::x(:,:)
! integer(4),intent(in)::n1(1_8:3_8)
! integer(4),intent(in)::n2(:)
! end
! end
