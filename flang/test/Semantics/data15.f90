! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Verify initialization extension: integer with logical, logical with integer
! CHECK: d (InDataStmt) size=20 offset=40: ObjectEntity type: LOGICAL(4) shape: 1_8:5_8 init:[LOGICAL(4)::transfer(-2_8,.false._4),transfer(-1_8,.false._4),.false._4,.true._4,transfer(2_8,.false._4)]
! CHECK: j (InDataStmt) size=8 offset=60: ObjectEntity type: INTEGER(4) shape: 1_8:2_8 init:[INTEGER(4)::0_4,1_4]
! CHECK: x, PARAMETER size=20 offset=0: ObjectEntity type: LOGICAL(4) shape: 1_8:5_8 init:[LOGICAL(4)::transfer(-2_8,.false._4),transfer(-1_8,.false._4),.false._4,.true._4,transfer(2_8,.false._4)]
! CHECK: y, PARAMETER size=20 offset=20: ObjectEntity type: INTEGER(4) shape: 1_8:5_8 init:[INTEGER(4)::-2_4,-1_4,0_4,1_4,2_4]
program main
  logical, parameter :: x(5) = [ -2, -1, 0, 1, 2 ]
  integer, parameter :: y(5) = x
  logical :: d(5)
  integer :: j(2)
  data d / -2, -1, 0, 1, 2 /
  data j / .false., .true. /
end

