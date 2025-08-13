!RUN: %flang_fc1 -cpp -fdebug-unparse %s | FileCheck %s
PROGRAM P
#if (1 && 2)
  !CHECK: TRUE
  WRITE(*,*) 'TRUE'
#else
  !CHECK-NOT: FALSE
  WRITE(*,*) 'FALSE'
#endif
#if ((1 || 2) != 3)
  !CHECK: TRUE
  WRITE(*,*) 'TRUE'
#else
  !CHECK-NOT: FALSE
  WRITE(*,*) 'FALSE'
#endif
END PROGRAM

