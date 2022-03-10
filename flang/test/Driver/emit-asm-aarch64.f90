! Test -S (AArch64)

! REQUIRES: aarch64-registered-target

!-------------
! RUN COMMANDS
!-------------
! RUN: %flang_fc1 -S -triple aarch64-unknown-linux-gnu %s -o - | FileCheck %s
! RUN: %flang -S -target aarch64-unknown-linux-gnu %s -o - | FileCheck %s

!----------------
! EXPECTED OUTPUT
!----------------
! CHECK-LABEL: _QQmain:
! CHECK-NEXT: .Lfunc_begin0:
! CHECK: ret

!------
! INPUT
!------
end program
