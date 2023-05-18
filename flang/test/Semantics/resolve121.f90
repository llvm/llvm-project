! Check that symbols without SAVE attribute from an EQUIVALENCE
! with at least one symbol being SAVEd (explicitly or implicitly)
! have implicit SAVE attribute.
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

subroutine test1()
  ! CHECK-LABEL: Subprogram scope: test1
  ! CHECK: i1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4) init:1_4
  ! CHECK: j1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4)
  integer :: i1 = 1
  integer :: j1
  equivalence(i1,j1)
end subroutine test1

subroutine test2()
  ! CHECK-LABEL: Subprogram scope: test2
  ! CHECK: i1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4) init:1_4
  ! CHECK: j1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4)
  integer :: i1 = 1
  integer :: j1
  equivalence(j1,i1)
end subroutine test2

subroutine test3()
  ! CHECK-LABEL: Subprogram scope: test3
  ! CHECK: i1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4)
  ! CHECK: j1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4)
  ! CHECK: k1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4)
  integer :: i1
  integer :: j1, k1
  common /blk/ k1
  save /blk/
  equivalence(i1,j1,k1)
end subroutine test3

subroutine test4()
  ! CHECK-LABEL: Subprogram scope: test4
  ! CHECK: i1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4) init:1_4
  ! CHECK: j1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4)
  ! CHECK: k1, SAVE size=4 offset=0: ObjectEntity type: INTEGER(4)
  integer :: i1 = 1
  integer :: j1, k1
  common /blk/ k1
  equivalence(i1,j1,k1)
end subroutine test4
