! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK-NOT: stop
! #define KWM !, then KWM works as comment line initiator
#define KWM !
KWM   print *, 'pp129.F90 FAIL HARD!'; stop
      print *, 'pp129.F90 yes'
      end
