! RUN: rm -rf %S/multiple-input-files.txt  %S/Inputs/hello-world.txt

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! TEST 1: Both input files are processed (output is printed to stdout)
! RUN: %flang-new -test-io %s %S/Inputs/hello-world.f90 | FileCheck %s  -check-prefix=flang-new

! TEST 2: None of the files is processed (not possible to specify the output file when multiple input files are present)
! RUN: not %flang-new -test-io -o - %S/Inputs/hello-world.f90 %s  2>&1 | FileCheck %s -check-prefix=ERROR
! RUN: not %flang-new -test-io -o %t %S/Inputs/hello-world.f90 %s 2>&1 | FileCheck %s -check-prefix=ERROR

!----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!----------------------------------------
! TEST 3: Both input files are processed
! RUN: %flang-new -fc1 -test-io  %S/Inputs/hello-world.f90 %s 2>&1 \
! RUN:  && FileCheck %s --input-file=%S/multiple-input-files.txt -check-prefix=flang-new-FC1-OUTPUT1 

! TEST 4: Only the last input file is processed
! RUN: %flang-new -fc1 -test-io  %S/Inputs/hello-world.f90 %s -o %t 2>&1 \
! RUN:  && FileCheck %s --input-file=%t -check-prefix=flang-new-FC1-OUTPUT1

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! TEST 1: By default, `flang-new` prints the output from all input files to
! stdout
! flang-new-LABEL: Program arithmetic
! flang-new-NEXT:    Integer :: i, j
! flang-new-NEXT:    i = 2; j = 3; i= i * j;
! flang-new-NEXT:  End Program arithmetic
! flang-new-NEXT: !This is a test file with a hello world in Fortran
! flang-new-NEXT:program hello
! flang-new-NEXT:  implicit none
! flang-new-NEXT:  write(*,*) 'Hello world!'
! flang-new-NEXT:end program hello


! TEST 2: `-o` does not work for `flang-new` when multiple input files are present
! ERROR:error: cannot specify -o when generating multiple output files


! TEST 3 & TEST 4: Unless the output file is specified, `flang-new -fc1` generates one output file for every input file. If an
! output file is specified (with `-o`), then only the last input file is processed.
! flang-new-FC1-OUTPUT1-LABEL: Program arithmetic
! flang-new-FC1-OUTPUT1-NEXT:    Integer :: i, j
! flang-new-FC1-OUTPUT1-NEXT:    i = 2; j = 3; i= i * j;
! flang-new-FC1-OUTPUT1-NEXT:  End Program arithmetic
! flang-new-FC1-OUTPUT1-NEXT: !This is a test file with a hello world in Fortran
! flang-new-FC1-OUTPUT1-NEXT:program hello
! flang-new-FC1-OUTPUT1-NEXT:  implicit none
! flang-new-FC1-OUTPUT1-NEXT:  write(*,*) 'Hello world!'
! flang-new-FC1-OUTPUT1-NEXT:end program hello


Program arithmetic
  Integer :: i, j
  i = 2; j = 3; i= i * j;
End Program arithmetic
