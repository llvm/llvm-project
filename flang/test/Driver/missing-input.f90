! Test the behaviour of the driver when input is missing or is invalid. Note
! that with the compiler driver (flang), the input _has_ to be specified.
! Indeed, the driver decides what "job/command" to create based on the input
! file's extension. No input file means that it doesn't know what to do
! (compile?  preprocess? link?). The frontend driver (flang -fc1) simply
! assumes that "no explicit input == read from stdin"

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: not %flang  2>&1 | FileCheck %s --check-prefix=FLANG-NO-FILE
! RUN: not %flang %t.f90 2>&1 | FileCheck %s --check-prefix=FLANG-NONEXISTENT-FILE

!-----------------------------------------
! FLANG FRONTEND DRIVER (flang -fc1)
!-----------------------------------------
! RUN: not %flang_fc1 %t.f90 2>&1  | FileCheck %s --check-prefix=FLANG-FC1-NONEXISTENT-FILE
! RUN: not %flang_fc1 %S 2>&1  | FileCheck %s --check-prefix=FLANG-FC1-DIR

! FLANG-NO-FILE: flang{{.*}}: error: no input files

! FLANG-NONEXISTENT-FILE: flang{{.*}}: error: no such file or directory: {{.*}}
! FLANG-NONEXISTENT-FILE: flang{{.*}}: error: no input files

! FLANG-FC1-NONEXISTENT-FILE: error: {{.*}} does not exist
! FLANG-FC1-DIR: error: {{.*}} is not a regular file
