! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK-NOT: goto 2
* #define c 1, then use c as label in fixed-form
#define c 1
c     print *, 'pp042.F yes'; goto 2
      print *, 'pp042.F no'
2     continue
      end
