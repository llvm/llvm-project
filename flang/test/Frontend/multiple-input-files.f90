! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! NOTE: Use `-E` so that the compiler driver stops after the 1st compilation phase, preprocessing. That's all we need.

! TEST 1: Both input files are processed (output is printed to stdout)
! RUN: %flang-new -E -Xflang -test-io %s %S/Inputs/hello-world.f90 | FileCheck %s --match-full-lines -check-prefix=FLANG

! TEST 2: None of the files is processed (not possible to specify the output file when multiple input files are present)
! RUN: not %flang-new -E -Xflang -test-io -o - %S/Inputs/hello-world.f90 %s  2>&1 | FileCheck %s --match-full-lines -check-prefix=ERROR
! RUN: not %flang-new -E -Xflang -test-io -o %t %S/Inputs/hello-world.f90 %s 2>&1 | FileCheck %s --match-full-lines -check-prefix=ERROR

!----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!----------------------------------------
! TEST 3: Both input files are processed
! This particular test case generates output files in the same directory as the input files. We need to copy the input files into a
! temporary directory to avoid polluting the source directory.
! RUN: rm -rf %t-dir && mkdir -p %t-dir && cd %t-dir
! RUN: cp %s . && cp %S/Inputs/hello-world.f90 .
! RUN: %flang-new -fc1 -test-io  hello-world.f90 multiple-input-files.f90 2>&1 \
! RUN:  && FileCheck %s --input-file=multiple-input-files.txt --match-full-lines -check-prefix=FC1-OUTPUT1 \
! RUN:  && FileCheck %s --input-file=hello-world.txt --match-full-lines -check-prefix=FC1-OUTPUT2

! TEST 4: Only the last input file is processed
! RUN: %flang-new -fc1 -test-io %s %S/Inputs/hello-world.f90 -o %t 2>&1 \
! RUN:  && FileCheck %s --input-file=%t --match-full-lines -check-prefix=FC1-OUTPUT3

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! TEST 1: By default, `flang-new` prints the output from all input files to
! stdout
! FLANG-LABEL: Program arithmetic
! FLANG-NEXT:    Integer :: i, j
! FLANG-NEXT:    i = 2; j = 3; i= i * j;
! FLANG-NEXT:  End Program arithmetic
! FLANG-NEXT:!This is a test file with a hello world in Fortran
! FLANG-NEXT:program hello
! FLANG-NEXT:  implicit none
! FLANG-NEXT:  write(*,*) 'Hello world!'
! FLANG-NEXT:end program hello

! TEST 2: `-o` does not work for `flang-new` when multiple input files are present
! ERROR: flang-new: error: cannot specify -o when generating multiple output files

! TEST 3: The output file _was not_ specified - `flang-new -fc1` will process all
! input files and generate one output file for every input file.
! FC1-OUTPUT1-LABEL: Program arithmetic
! FC1-OUTPUT1-NEXT:    Integer :: i, j
! FC1-OUTPUT1-NEXT:    i = 2; j = 3; i= i * j;
! FC1-OUTPUT1-NEXT:  End Program arithmetic

! FC1-OUTPUT2-LABEL:!This is a test file with a hello world in Fortran
! FC1-OUTPUT2-NEXT:program hello
! FC1-OUTPUT2-NEXT:  implicit none
! FC1-OUTPUT2-NEXT:  write(*,*) 'Hello world!'
! FC1-OUTPUT2-NEXT:end program hello

! TEST 4: The output file _was_ specified - `flang-new -fc1` will process only
! the last input file and generate the corresponding output.
! FC1-OUTPUT3-LABEL:!This is a test file with a hello world in Fortran
! FC1-OUTPUT3-NEXT:program hello
! FC1-OUTPUT3-NEXT:  implicit none
! FC1-OUTPUT3-NEXT:  write(*,*) 'Hello world!'
! FC1-OUTPUT3-NEXT:end program hello


Program arithmetic
  Integer :: i, j
  i = 2; j = 3; i= i * j;
End Program arithmetic
