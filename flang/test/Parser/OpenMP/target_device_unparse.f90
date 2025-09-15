! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck --ignore-case %s
! Verifies the unparsing of the Openmp Target Device constructs
PROGRAM main
    USE OMP_LIB
    IMPLICIT NONE
    INTEGER:: X, Y
    INTEGER:: M = 1

!--------------------------------------------
! constant argument
!--------------------------------------------
!CHECK: !$OMP TARGET DEVICE(0)
!CHECK: !$OMP END TARGET
!$OMP TARGET DEVICE(0)
  M = M + 1
!$OMP END TARGET

!--------------------------------------------
! constant expression argument
!--------------------------------------------
!CHECK: !$OMP TARGET DEVICE(2+1)
!CHECK: !$OMP END TARGET
!$OMP TARGET DEVICE(2+1)
  M = M + 1
!$OMP END TARGET

!--------------------------------------------
! variable argument
!--------------------------------------------
!CHECK: !$OMP TARGET DEVICE(X)
!CHECK: !$OMP END TARGET
!$OMP TARGET DEVICE(X)
  M = M + 1
!$OMP END TARGET

!--------------------------------------------
! variable expression argument
!--------------------------------------------
!CHECK: !$OMP TARGET DEVICE(X-Y)
!CHECK: !$OMP END TARGET
!$OMP TARGET DEVICE(X-Y)
  M = M + 1
!$OMP END TARGET

!--------------------------------------------
! Ancestor followed by constant argument
!--------------------------------------------
!CHECK: !$OMP TARGET DEVICE(ANCESTOR: 0)
!CHECK: !$OMP END TARGET
!$OMP TARGET DEVICE(ANCESTOR: 0)
  M = M + 1
!$OMP END TARGET

!--------------------------------------------
! Device_Num followed by constant argument
!--------------------------------------------
!CHECK: !$OMP TARGET DEVICE(DEVICE_NUM: 1)
!CHECK: !$OMP END TARGET
!$OMP TARGET DEVICE(DEVICE_NUM: 1)
  M = M + 1
!$OMP END TARGET

!--------------------------------------------
! Ancestor followed by variable expression argument
!--------------------------------------------
!CHECK: !$OMP TARGET DEVICE(ANCESTOR: X+Y)
!CHECK: !$OMP END TARGET
!$OMP TARGET DEVICE(ANCESTOR: X + Y)
  M = M + 1
!$OMP END TARGET

!--------------------------------------------
! Device_Num followed by variable expression argument
!--------------------------------------------
!CHECK: !$OMP TARGET DEVICE(DEVICE_NUM: X-Y)
!CHECK: !$OMP END TARGET
!$OMP TARGET DEVICE(DEVICE_NUM: X - Y)
  M = M + 1
!$OMP END TARGET
END PROGRAM
