! RUN: %flang -mtune=pentium4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK: attributes #0 = {
!CHECK: "tune-cpu"="pentium4"
subroutine a
end subroutine a
