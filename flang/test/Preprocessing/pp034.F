! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if (777 .eq. 777) then
! CHECK: print *, 'pp034.F no: ', 777
* \ newline allowed in name on KWM definition
      integer, parameter :: KWMC = 666
#define KWM\
C 777
      if (KWMC .eq. 777) then
        print *, 'pp034.F yes'
      else
        print *, 'pp034.F no: ', KWMC
      end if
      end
