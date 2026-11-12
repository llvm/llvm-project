! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK-NOT: FAIL HARD!
* #define KWM c, then KWM works as comment line initiator
#define KWM c
KWM   print *, 'pp040.F FAIL HARD!'; stop
      print *, 'pp040.F yes'
      end
