! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: module attributes {
! CHECK-SAME: llvm.ident = "{{.*}}flang version {{.+}}"
