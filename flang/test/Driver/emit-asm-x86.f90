! Test -S (X86)

! REQUIRES: x86-registered-target

!-------------
! RUN COMMANDS
!-------------
! RUN: %flang_fc1 -S -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s
! RUN: %flang -S -target x86_64-unknown-linux-gnu %s -o - | FileCheck %s

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
