! Check that LLVM IR input is passed to clang instead of flang1.

! REQUIRES: classic_flang
! RUN: %clang --driver-mode=flang -S %S/Inputs/llvm-ir-input.ll -### 2>&1 | FileCheck %s

! CHECK-NOT: flang1
! CHECK: "{{.*}}clang{{.*}}" "-cc1"
