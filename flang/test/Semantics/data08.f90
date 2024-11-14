! RUN: %flang_fc1 -fdebug-dump-symbols -pedantic %s 2>&1 | FileCheck %s \
! RUN:   --check-prefixes=%if target={{.*-aix.*|sparc.*}} %{"CHECK","BE"%} \
! RUN:                    %else %{"CHECK","LE"%}

! CHECK: DATA statement value initializes 'jx' of type 'INTEGER(4)' with CHARACTER
! CHECK: DATA statement value initializes 'jy' of type 'INTEGER(4)' with CHARACTER
! CHECK: DATA statement value initializes 'jz' of type 'INTEGER(4)' with CHARACTER
! CHECK: DATA statement value initializes 'kx' of type 'INTEGER(8)' with CHARACTER
! LE: jx (InDataStmt) size=4 offset=0: ObjectEntity type: INTEGER(4) init:1684234849_4
! BE: jx (InDataStmt) size=4 offset=0: ObjectEntity type: INTEGER(4) init:1633837924_4
! LE: jy (InDataStmt) size=4 offset=4: ObjectEntity type: INTEGER(4) init:543384161_4
! BE: jy (InDataStmt) size=4 offset=4: ObjectEntity type: INTEGER(4) init:1633837856_4
! LE: jz (InDataStmt) size=4 offset=8: ObjectEntity type: INTEGER(4) init:1684234849_4
! BE: jz (InDataStmt) size=4 offset=8: ObjectEntity type: INTEGER(4) init:1633837924_4
! LE: kx (InDataStmt) size=8 offset=16: ObjectEntity type: INTEGER(8) init:7523094288207667809_8
! BE: kx (InDataStmt) size=8 offset=16: ObjectEntity type: INTEGER(8) init:7017280452245743464_8

integer :: jx, jy, jz
integer(8) :: kx
data jx/4habcd/
data jy/3habc/
data jz/5habcde/
data kx/'abcdefgh'/
end
