// XFAIL: *

! REQUIRES: classic_flang

! Check that the -emit-flang-llvm option dumps LLVM IR pre-optimisation

! RUN: %clang --driver-mode=flang -emit-flang-llvm -S -o %t.ll %s -### 2>&1 \
! RUN:   | FileCheck %s
! CHECK-NOT: argument unused during compilation: '-S'
! CHECK: "{{.*}}flang1"
! CHECK-NEXT: "{{.*}}flang2"
! CHECK-NOT: "{{.*}}clang{{.*}}" "-cc1"
