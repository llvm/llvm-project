! RUN: rm -rf %S/input-output-file.txt

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! TEST 1: Print to stdout (implicit)
! RUN: %flang-new -test-io %s  2>&1 | FileCheck %s --match-full-lines
! TEST 2: Print to stdout (explicit)
! RUN: %flang-new -test-io -o - %s  2>&1 | FileCheck %s --match-full-lines
! TEST 3: Print to a file
! RUN: %flang-new -test-io -o %t %s 2>&1 && FileCheck %s --match-full-lines --input-file=%t

!----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!----------------------------------------
! TEST 4: Write to a file (implicit)
! RUN: %flang-new -fc1 -test-io  %s 2>&1 && FileCheck %s --match-full-lines --input-file=%S/input-output-file.txt
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
