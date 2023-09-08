!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

! Size and alignment with EQUIVALENCE and COMMON

! a1 depends on a2 depends on a3
module ma
  real :: a1(10), a2(10), a3(10)
  equivalence(a1, a2(3)) !CHECK: a1, PUBLIC size=40 offset=20:
  equivalence(a2, a3(4)) !CHECK: a2, PUBLIC size=40 offset=12:
  !CHECK: a3, PUBLIC size=40 offset=0:
end

! equivalence and 2-dimensional array
module mb
  real :: b1(4), b2, b3, b4
  real :: b(-1:1,2:6)     !CHECK: b, PUBLIC size=60 offset=0:
  equivalence(b(1,6), b1) !CHECK: b1, PUBLIC size=16 offset=56:
  equivalence(b(1,5), b2) !CHECK: b2, PUBLIC size=4 offset=44:
  equivalence(b(0,6), b3) !CHECK: b3, PUBLIC size=4 offset=52:
  equivalence(b(0,4), b4) !CHECK: b4, PUBLIC size=4 offset=28:
end

! equivalence and substring
subroutine mc         !CHECK: Subprogram scope: mc size=12 alignment=1
  character(10) :: c1 !CHECK: c1 size=10 offset=0:
  character(5)  :: c2 !CHECK: c2 size=5 offset=7:
  equivalence(c1(9:), c2(2:4))
end

! Common block: objects are in order from COMMON statement and not part of module
module md                   !CHECK: Module scope: md size=1 alignment=1
  integer(1) :: i
  integer(2) :: d1          !CHECK: d1, PUBLIC size=2 offset=8:
  integer(4) :: d2          !CHECK: d2, PUBLIC size=4 offset=4:
  integer(1) :: d3          !CHECK: d3, PUBLIC size=1 offset=0:
  real(2) :: d4             !CHECK: d4, PUBLIC size=2 offset=0:
  common /common1/ d3,d2,d1 !CHECK: common1 size=10 offset=0: CommonBlockDetails alignment=4:
  common /common2/ d4       !CHECK: common2 size=2 offset=0: CommonBlockDetails alignment=2:
end

! Test extension of common block size through equivalence statements.
module me
  integer :: i1, j1, l1(10)
  equivalence(i1, l1)
  common /common3/ j1, i1   ! CHECK: common3 size=44 offset=0: CommonBlockDetails alignment=4:

  integer :: i2, j2, l2(10)
  equivalence(i2, l2(2))
  common /common4/ j2, i2   ! CHECK: common4 size=40 offset=0: CommonBlockDetails alignment=4:

  integer :: i3, j3, l3(10)
  equivalence(i3, l3)
  common /common5/ i3, j3   ! CHECK: common5 size=40 offset=0: CommonBlockDetails alignment=4:

  integer :: i4, j4, l4(10), k4(10)
  equivalence(i4, l4)
  equivalence(l4(10), k4)
  common /common6/ i4, j4   ! CHECK: common6 size=76 offset=0: CommonBlockDetails alignment=4:

  integer :: i5, j5, l5(10)
  equivalence(l5(1), i5)
  common /common7/ j5, i5   ! CHECK: common7 size=44 offset=0: CommonBlockDetails alignment=4:

  real :: a1, a2, a3(2)
  equivalence(a1,a3(1)),(a2,a3(2))
  common /common8/ a1, a2   ! CHECK: common8 size=8 offset=0: CommonBlockDetails alignment=4:
end

! Ensure EQUIVALENCE of undeclared names in internal subprogram doesn't
! attempt to host-associate
subroutine host1
 contains
  subroutine internal
    common /b/ x(4)  ! CHECK: x (Implicit) size=16 offset=0: ObjectEntity type: REAL(4) shape: 1_8:4_8
    equivalence(x,y) ! CHECK: y (Implicit) size=4 offset=0: ObjectEntity type: REAL(4)
  end
end
