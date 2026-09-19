!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!Regression test for bug #119005
character*2 :: ary4
!CHECK: DATA ary4/"cd"/
data ary4/"abcdef"(3:4)/
end

! Regression test: constant expressions (not just literals) must be accepted
! in old-style slash initialization.
subroutine test_slash_init_const_expr
!CHECK: INTEGER int1/4_4/
  integer :: int1 /2**2/
!CHECK: INTEGER int2/2_4/
  integer :: int2 /2/
  print*, int1, int2
end subroutine
