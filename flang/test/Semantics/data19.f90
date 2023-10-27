! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Test truncation/padding in DATA statement.

  character(len=3) :: c1, c2, c3(2), c4(2)
  data c1(1:2), c1(3:3) /'123', '4'/
  data c2(1:2), c2(3:3) /'1', '2'/
  data c3(:)(1:2), c3(:)(3:3) /'123', '678', '4', '9'/
  data c4(:)(1:2), c4(:)(3:3) /'1', '6', '2', '7'/
end
!CHECK:  c1 (InDataStmt) size=3 offset=0: ObjectEntity type: CHARACTER(3_4,1) init:"124"
!CHECK:  c2 (InDataStmt) size=3 offset=3: ObjectEntity type: CHARACTER(3_4,1) init:"1 2"
!CHECK:  c3 (InDataStmt) size=6 offset=6: ObjectEntity type: CHARACTER(3_4,1) shape: 1_8:2_8 init:[CHARACTER(KIND=1,LEN=3)::"124","679"]
!CHECK:  c4 (InDataStmt) size=6 offset=12: ObjectEntity type: CHARACTER(3_4,1) shape: 1_8:2_8 init:[CHARACTER(KIND=1,LEN=3)::"1 2","6 7"]
