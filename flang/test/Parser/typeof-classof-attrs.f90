! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-unparse-no-sema %s > %t.f90
! RUN: %flang_fc1 -fdebug-unparse-no-sema %t.f90 2>&1 | FileCheck %s
! Test TYPEOF and CLASSOF with spaces and attributes.

program test_program
  implicit none

  TYPE :: matrix
    INTEGER :: v
    !CHECK: INTEGER, POINTER, PROTECTED_TARGET :: readonly
    INTEGER, POINTER, PROTECTED_TARGET :: readonly
  END TYPE

  TYPE(matrix) :: MAT

  !CHECK: TYPEOF(mat), POINTER :: tmat_ptr
  TYPEOF(MAT), POINTER :: TMAT_PTR
  !CHECK: TYPEOF(mat), ALLOCATABLE, TARGET :: tmat_allocatable
  TYPEOF(MAT), ALLOCATABLE, TARGET :: TMAT_ALLOCATABLE

  !CHECK: CLASSOF(mat), POINTER :: cmat_ptr
  CLASSOF(MAT), POINTER :: CMAT_PTR
  !CHECK: CLASSOF(mat), ALLOCATABLE, TARGET :: cmat_allocatable
  CLASSOF(MAT), ALLOCATABLE, TARGET :: CMAT_ALLOCATABLE

  !CHECK: TYPEOF(mat), POINTER, PROTECTED_TARGET :: tmat_readonly
  TYPEOF(MAT), POINTER, PROTECTED_TARGET :: TMAT_READONLY
  !CHECK: CLASSOF(mat), PROTECTED_TARGET, POINTER :: cmat_readonly
  CLASSOF(MAT), PROTECTED_TARGET, POINTER :: CMAT_READONLY
  INTEGER, POINTER :: p1, p2, p3
  !CHECK: PROTECTED_TARGET :: p1, p2
  PROTECTED_TARGET p1, p2
  !CHECK: PROTECTED_TARGET :: p3
  PROTECTED_TARGET :: p3
end program
