! RUN: %flang_fc1 -fsyntax-only -pedantic %s  2>&1 | FileCheck %s --allow-empty
! Check subroutine with alt return

       SUBROUTINE TEST (N, *, *)
       IF ( N .EQ. 0 ) RETURN
       IF ( N .EQ. 1 ) RETURN 1
       RETURN 2
       END
! CHECK-NOT: error:
! CHECK-NOT: portability:
