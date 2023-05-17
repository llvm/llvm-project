! Verify that reading from stdin works as expected - Fortran input

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! Input type is implicit
! RUN: cat %s | %flang -E -cpp - | FileCheck %s --check-prefix=PP-NOT-DEFINED
! RUN: cat %s | %flang -DNEW -E -cpp - | FileCheck %s --check-prefix=PP-DEFINED
! RUN: cat %s | %flang -DNEW -E - | FileCheck %s --check-prefix=PP-NOT-DEFINED
! RUN: cat %s | %flang -DNEW -E -nocpp - | FileCheck %s --check-prefix=PP-NOT-DEFINED

! Input type is explicit
! RUN: cat %s | %flang -E -cpp -x f95-cpp-input - | FileCheck %s --check-prefix=PP-NOT-DEFINED
! RUN: cat %s | %flang -DNEW -E -cpp -x f95-cpp-input - | FileCheck %s --check-prefix=PP-DEFINED

!---------------------------------------
! FLANG FRONTEND DRIVER (flang -fc1)
!---------------------------------------
! Test `-E`: for the corresponding frontend actions the driver relies on the prescanner API to handle file I/O
! RUN: cat %s | %flang -fc1 -E -cpp | FileCheck %s --check-prefix=PP-NOT-DEFINED
! RUN: cat %s | %flang -fc1 -DNEW -E -cpp | FileCheck %s --check-prefix=PP-DEFINED

! Test `-test-io`: for the corresponding frontend action (`InputOutputTestAction`) the driver handles the file I/O on its own
! the corresponding action (`PrintPreprocessedAction`)
! RUN: cat %s | %flang -fc1 -test-io -cpp | FileCheck %s --check-prefix=IO --match-full-lines
! RUN: cat %s | %flang -fc1 -DNEW -cpp -test-io | FileCheck %s --check-prefix=IO --match-full-lines

! PP-NOT-DEFINED: Program B
! PP-DEFINED: Program A

! IO: #ifdef NEW
! IO-NEXT:   Program A
! IO-NEXT: #else
! IO-NEXT:   Program B
! IO-NEXT: #endif

#ifdef NEW
  Program A
#else
  Program B
#endif
