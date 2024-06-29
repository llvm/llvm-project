! RUN: %flang -mtune=pentium4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK: attributes #{{[0-9]+}} = {
!CHECK-SAME: "tune-cpu"="pentium4"
subroutine a
end subroutine a
