! RUN: %f18 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! CHECK: DATA statement value initializes 'jx' of type 'INTEGER(4)' with CHARACTER
! CHECK: DATA statement value initializes 'jy' of type 'INTEGER(4)' with CHARACTER
! CHECK: DATA statement value initializes 'jz' of type 'INTEGER(4)' with CHARACTER
! CHECK: DATA statement value initializes 'kx' of type 'INTEGER(8)' with CHARACTER
! CHECK: jx (InDataStmt) size=4 offset=0: ObjectEntity type: INTEGER(4) init:1684234849_4
! CHECK: jy (InDataStmt) size=4 offset=4: ObjectEntity type: INTEGER(4) init:543384161_4
! CHECK: jz (InDataStmt) size=4 offset=8: ObjectEntity type: INTEGER(4) init:1684234849_4
! CHECK: kx (InDataStmt) size=8 offset=16: ObjectEntity type: INTEGER(8) init:7523094288207667809_8

integer :: jx, jy, jz
integer(8) :: kx
data jx/4habcd/
data jy/3habc/
data jz/5habcde/
data kx/'abcdefgh'/
end
