! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Verify that allocatable components have explicit NULL() initializers
! when converted from DATA statements
!CHECK: x1 (InDataStmt) size=32 offset=0: ObjectEntity type: TYPE(t) init:t(a=NULL(),n=1_4)
!CHECK: x2 (InDataStmt) size=64 offset=32: ObjectEntity type: TYPE(t) shape: 1_8:2_8 init:[t::t(a=NULL(),n=2_4),t(a=NULL(),n=3_4)]
!CHECK: x3 (InDataStmt) size=64 offset=96: ObjectEntity type: TYPE(t2) init:t2(b=[t::t(a=NULL(),n=4_4),t(a=NULL(),n=5_4)])
program main
  type t
    real, allocatable :: a
    integer n
  end type
  type t2
    type(t) b(2)
  end type
  type(t) x1, x2(2)
  type(t2) x3
  data x1%n/1/, x2(:)%n/2, 3/, x3%b(:)%n/4, 5/
end
