! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! Test TYPEOF and CLASSOF with spaces and attributes.

program test_program
  implicit none

  TYPE :: matrix
    INTEGER :: v
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
end program
