! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Verify that the closure of EQUIVALENCE'd symbols with any DATA
! initialization produces a combined initializer, with explicit
! initialization overriding any default component initialization.
! CHECK: .F18.0, SAVE (CompilerCreated) size=8 offset=0: ObjectEntity type: INTEGER(4) shape: 1_8:2_8 init:[INTEGER(4)::456_4,234_4]
! CHECK: ja, SAVE (InDataStmt) size=8 offset=0: ObjectEntity type: INTEGER(4) shape: 1_8:2_8
! CHECK-NOT: x0, SAVE size=8 offset=8: ObjectEntity type: TYPE(t1) init:t1(m=123_4,n=234_4)
! CHECK: x1 size=8 offset=16: ObjectEntity type: TYPE(t1) init:t1(m=345_4,n=234_4)
! CHECK: x2, SAVE size=8 offset=0: ObjectEntity type: TYPE(t1)
! CHECK-NOT: x3a, SAVE size=8 offset=24: ObjectEntity type: TYPE(t3) init:t3(t2=t2(k=567_4),j=0_4)
! CHECK: x3b size=8 offset=32: ObjectEntity type: TYPE(t3) init:t3(k=567_4,j=678_4)
! CHECK: Equivalence Sets: (x2,ja(1)) (.F18.0,x2)
type :: t1
  sequence
  integer :: m = 123
  integer :: n = 234
end type
type :: t2
  integer :: k = 567
end type
type, extends(t2) :: t3
  integer :: j ! uninitialized
end type
type(t1), save :: x0 ! not enabled
type(t1) :: x1 = t1(m=345)
type(t1) :: x2
type(t3), save :: x3a ! not enabled
type(t3) :: x3b = t3(j=678)
integer :: ja(2)
equivalence(x2, ja)
data ja(1)/456/
end
