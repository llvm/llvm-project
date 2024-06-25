! REQUIRES: aarch64-registered-target
! RUN: %flang -mtune=neoverse-n1 -S -emit-llvm %s -o - | FileCheck %s

!CHECK: attributes #{{[0-9]+}} = {
!CHECK-SAME: "tune-cpu"="neoverse-n1"
subroutine a
end subroutine a
