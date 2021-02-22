! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! NOTE: Use `-E` so that the compiler driver stops after the 1st compilation phase, preprocessing. That's all we need.

! TEST 1: Print to stdout (implicit)
! RUN: %flang-new -E -Xflang -test-io %s  2>&1 | FileCheck %s --match-full-lines

! TEST 2: Print to stdout (explicit)
! RUN: %flang-new -E -Xflang -test-io -o - %s  2>&1 | FileCheck %s --match-full-lines

! TEST 3: Print to a file
! RUN: %flang-new -E -Xflang -test-io -o %t %s 2>&1 && FileCheck %s --match-full-lines --input-file=%t

!----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!----------------------------------------
! TEST 4: Write to a file (implicit)
! This particular test case generates an output file in the same directory as the input file. We need to copy the input file into a
! temporary directory to avoid polluting the source directory.
! RUN: rm -rf %t-dir && mkdir -p %t-dir && cd %t-dir
! RUN: cp %s .
! RUN: %flang-new -fc1 -test-io input-output-file.f90  2>&1 && FileCheck %s --match-full-lines --input-file=input-output-file.txt

! TEST 5: Write to a file (explicit)
! RUN: %flang-new -fc1 -test-io  -o %t %s 2>&1 && FileCheck %s --match-full-lines --input-file=%t

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! CHECK-LABEL: Program arithmetic
! CHECK-NEXT:    Integer :: i, j
! CHECK-NEXT:    i = 2; j = 3; i= i * j;
! CHECK-NEXT:  End Program arithmetic

Program arithmetic
  Integer :: i, j
  i = 2; j = 3; i= i * j;
End Program arithmetic
