! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! REAL(3) is bfloat16: its raw bits occupy 2 bytes, not 3.  Constant
! initialization and TRANSFER round-trip values through their raw bytes, so
! the number of bytes stored must not be conflated with the kind number.
! See fold-real10-storage-size.f90 for the REAL(10) counterpart.

subroutine data_init
  real(3) :: x
  data x/1.5_3/
  print *, x
end subroutine
! CHECK-LABEL: SUBROUTINE data_init
! CHECK: DATA x/1.5_3/

subroutine transfers
  print *, transfer(1.5_3, 0.0_3)
  print *, transfer(1.5_3, 0_2)
  print *, transfer(16320_2, 0.0_3)
end subroutine
! CHECK-LABEL: SUBROUTINE transfers
! CHECK: PRINT *, 1.5_3
! CHECK: PRINT *, 16320_2
! CHECK: PRINT *, 1.5_3
