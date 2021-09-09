! Test the behaviour of the driver when input is missing or is invalid. Note
! that with the compiler driver (flang-new), the input _has_ to be specified.
! Indeed, the driver decides what "job/command" to create based on the input
! file's extension. No input file means that it doesn't know what to do
! (compile?  preprocess? link?). The frontend driver (flang-new -fc1) simply
! assumes that "no explicit input == read from stdin"

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang  2>&1 | FileCheck %s --check-prefix=FLANG-NO-FILE
! RUN: not %flang %t 2>&1 | FileCheck %s --check-prefix=FLANG-NONEXISTENT-FILE

!-----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: not %flang_fc1 %t 2>&1  | FileCheck %s --check-prefix=FLANG-FC1-NONEXISTENT-FILE
! RUN: not %flang_fc1 %S 2>&1  | FileCheck %s --check-prefix=FLANG-FC1-DIR

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! FLANG-NO-FILE: flang-new: error: no input files

! FLANG-NONEXISTENT-FILE: flang-new: error: no such file or directory: {{.*}}
! FLANG-NONEXISTENT-FILE: flang-new: error: no input files

! FLANG-FC1-NONEXISTENT-FILE: error: {{.*}} does not exist
! FLANG-FC1-DIR: error: {{.*}} is not a regular file
