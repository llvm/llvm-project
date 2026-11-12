! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Ensure that implicitly typed DATA statement objects with derived
! types get their symbols resolved by the end of the name resolution pass.
! CHECK: x1 (Implicit, InDataStmt) size=4 offset=0: ObjectEntity type: TYPE(t1) shape: 1_8:1_8 init:[t1::t1(n=123_4)]
! CHECK: x2 (InDataStmt) size=4 offset=4: ObjectEntity type: TYPE(t2) shape: 1_8:1_8 init:[t2::t2(m=456_4)]
implicit type(t1)(x)
type t1
  integer n
end type
dimension x1(1), x2(1)
data x1(1)%n /123/
data x2(1)%m /456/
type t2
  integer m
end type
type(t2) x2
end
