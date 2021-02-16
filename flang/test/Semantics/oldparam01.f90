! RUN: %f18 -falternative-parameter-statement -fdebug-dump-symbols %s 2>&1 | FileCheck %s

! Non-error tests for "old style" PARAMETER statements

type :: t
  integer(kind=4) :: n
end type
!CHECK: x1, PARAMETER size=4 offset=0: ObjectEntity type: INTEGER(4) init:1_4
parameter x1 = 1_4 ! integer scalar
!CHECK: x2, PARAMETER size=4 offset=4: ObjectEntity type: INTEGER(4) shape: 1_8:1_8 init:[INTEGER(4)::2_4]
parameter x2 = [2_4] ! integer vector
!CHECK: x3, PARAMETER size=4 offset=8: ObjectEntity type: TYPE(t) init:t(n=3_4)
parameter x3 = t(3) ! derived scalar
!CHECK: x4, PARAMETER size=8 offset=12: ObjectEntity type: TYPE(t) shape: 1_8:2_8 init:[t::t(n=4_4),t(n=5_4)]
parameter x4 = [t(4), t(5)] ! derived vector
!CHECK: x5, PARAMETER size=3 offset=20: ObjectEntity type: CHARACTER(3_8,1) init:"abc"
parameter x5 = 1_"abc" ! character scalar
!CHECK: x6, PARAMETER size=12 offset=23: ObjectEntity type: CHARACTER(4_8,1) shape: 1_8:3_8 init:[CHARACTER(KIND=1,LEN=4)::"defg","h   ","ij  "]
parameter x6 = [1_"defg", 1_"h", 1_"ij"] ! character scalar
!CHECK: x7, PARAMETER size=4 offset=36: ObjectEntity type: INTEGER(4) init:5_4
!CHECK: x8, PARAMETER size=4 offset=40: ObjectEntity type: INTEGER(4) init:4_4
parameter x7 = 2+3, x8 = 4 ! folding, multiple definitions
!CHECK: x9, PARAMETER size=4 offset=44: ObjectEntity type: LOGICAL(4) init:.true._4
parameter x9 = .true.
end
