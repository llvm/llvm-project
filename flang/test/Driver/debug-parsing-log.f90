! RUN: %flang_fc1 -fdebug-dump-parsing-log %s  2>&1 | FileCheck %s

! Below are just few lines extracted from the dump. The actual output is much _much_ bigger.

! CHECK: {{.*[/\\]}}debug-parsing-log.f90:25:1: IMPLICIT statement
! CHECK-NEXT:  END PROGRAM
! CHECK-NEXT:  ^
! CHECK-NEXT:  fail 3
! CHECK-NEXT: {{.*[/\\]}}debug-parsing-log.f90:25:1: error: expected 'IMPLICIT NONE'
! CHECK-NEXT:   END PROGRAM
! CHECK-NEXT:   ^
! CHECK-NEXT: {{.*[/\\]}}debug-parsing-log.f90:25:1: in the context: IMPLICIT statement
! CHECK-NEXT:   END PROGRAM
! CHECK-NEXT:   ^
! CHECK-NEXT: {{.*[/\\]}}debug-parsing-log.f90:25:1: in the context: implicit part
! CHECK-NEXT:   END PROGRAM
! CHECK-NEXT:   ^
! CHECK-NEXT: {{.*[/\\]}}debug-parsing-log.f90:25:1: in the context: specification part
! CHECK-NEXT:   END PROGRAM
! CHECK-NEXT:   ^
! CHECK-NEXT: {{.*[/\\]}}debug-parsing-log.f90:25:1: in the context: main program
! CHECK-NEXT:   END PROGRAM
! CHECK-NEXT:   ^

END PROGRAM
