! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Ensure that DATA-style default component /initializers/ are processed
! before they are needed to handle EQUIVALENCE'd storage.
type t
  sequence
  integer :: j(10) /1,2,3,4,5,6,7,8,9,10/
end type
type(t) :: A
integer arr(10)
equivalence (A, arr)
end

!CHECK: .F18.0, SAVE (CompilerCreated) size=40 offset=0: ObjectEntity type: INTEGER(4) shape: 1_8:10_8 init:[INTEGER(4)::1_4,2_4,3_4,4_4,5_4,6_4,7_4,8_4,9_4,10_4]
!CHECK: a size=40 offset=0: ObjectEntity type: TYPE(t)
!CHECK: arr size=40 offset=0: ObjectEntity type: INTEGER(4) shape: 1_8:10_8
!CHECK: Equivalence Sets: (a,arr(1)) (.F18.0,a)
