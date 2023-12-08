! Verify that the driver correctly processes `-fsyntax-only`.
!
! By default, the compiler driver (`flang`) will create actions/phases to
! generate machine code (i.e. object files). The `-fsyntax-only` flag is a
! "phase-control" flag that controls this behavior and makes the driver stop
! once the semantic checks have been run. The frontend driver (`flang -fc1`)
! runs `-fsyntax-only` by default (i.e. that's the default action), so the flag
! can be skipped.

! RUN: %flang -fsyntax-only %s 2>&1 | FileCheck %s --allow-empty
! RUN: %flang_fc1 %s 2>&1 | FileCheck %s --allow-empty

! RUN: rm -rf %t/non-existent-dir/
! RUN: not %flang -c %s -o %t/non-existent-dir/syntax-only.o 2>&1 | FileCheck %s --check-prefix=NO_FSYNTAX_ONLY
! RUN: not %flang_fc1 -emit-obj %s -o %t/non-existent-dir/syntax-only.o 2>&1 | FileCheck %s --check-prefix=NO_FSYNTAX_ONLY

! CHECK-NOT: error
! NO_FSYNTAX_ONLY: error: failed to create the output file

end program
