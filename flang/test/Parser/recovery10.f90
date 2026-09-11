! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s \
! RUN:   --implicit-check-not="expected 'CALL'" \
! RUN:   --implicit-check-not="expected 'DATA'" \
! RUN:   --implicit-check-not="expected 'NAMELIST'" \
! RUN:   --implicit-check-not="expected 'GO TO'" \
! RUN:   --implicit-check-not="expected 'ALLOCATE ('" \
! RUN:   --implicit-check-not="obsolete legacy extension"

! Verify that an unrecognizable statement in the execution part produces a
! single clear "expected an executable statement" diagnostic instead of a
! flood of "expected 'KEYWORD'" messages for every possible statement, one
! for each execution-part-construct alternative.  The --implicit-check-not
! options assert, across the whole output, that none of those spurious
! messages reappear.

program p
  x = 1 ;  + y
! CHECK: error: expected an executable statement
! CHECK-NEXT: {{.*}}x = 1 ;  + y
! CHECK: in the context: execution part
end
