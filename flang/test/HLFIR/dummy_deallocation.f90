! RUN: bbc -emit-fir -hlfir %s -o - | FileCheck %s
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test that the intent(out) allocatable dummy argument
! is not deallocated in entry SUB_B.

! CHECK-LABEL: func.func @_QPsub_a
! CHECK: fir.freemem

! CHECK-LABEL: func.func @_QPsub_b
! CHECK-NOT: fir.freemem
SUBROUTINE SUB_A(A)
  INTEGER, INTENT(out), ALLOCATABLE, DIMENSION (:) :: A
  RETURN
  ENTRY SUB_B
END SUBROUTINE SUB_A
