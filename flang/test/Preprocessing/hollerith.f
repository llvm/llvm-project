! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: character*1hi
! CHECK: dataa/1*1h /
! CHECK: datab/1*1h /
! CHECK: do1h=1,2
      CHARACTER*1H I
      CHARACTER*1 A,B
      INTEGER H
      DATA A/1*1H /
      DATA B/
     +1*1H /
      DO1H =1,2
 1    CONTINUE
      END
