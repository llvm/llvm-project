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
