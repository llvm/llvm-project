! This test verifies the phase control in Flang compiler driver.

! RUN: %flang -E -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=PP
! RUN: %flang -fsyntax-only -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=COMPILE
! RUN: %flang -c -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=EMIT_OBJ

! PP: +- 0: input, "{{.*}}phases.f90", f95-cpp-input
! PP-NEXT: 1: preprocessor, {0}, f95

! COMPILE: +- 0: input, "{{.*}}phases.f90", f95-cpp-input
! COMPILE-NEXT: 1: preprocessor, {0}, f95
! COMPILE-NEXT: 2: compiler, {1}, none

! EMIT_OBJ: +- 0: input, "{{.*}}phases.f90", f95-cpp-input
! EMIT_OBJ-NEXT: 1: preprocessor, {0}, f95
! EMIT_OBJ-NEXT: 2: compiler, {1}, ir
! EMIT_OBJ-NEXT: +- 3: backend, {2}, assembler
! EMIT_OBJ-NEXT: 4: assembler, {3}, object
