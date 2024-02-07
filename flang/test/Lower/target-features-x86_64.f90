! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=ALL,NONE
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o - | FileCheck %s --check-prefixes=ALL,CPU
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-feature +sse %s -o - | FileCheck %s --check-prefixes=ALL,FEATURE
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-cpu x86-64 -target-feature +sse %s -o - | FileCheck %s --check-prefixes=ALL,BOTH

! ALL: module attributes {

! NONE-NOT: fir.target_cpu
! NONE-NOT: fir.target_features

! CPU-SAME: fir.target_cpu = "x86-64"
! CPU-NOT: fir.target_features

! FEATURE-NOT: fir.target_cpu
! FEATURE-SAME: fir.target_features = #llvm.target_features<["+sse"]>

! BOTH-SAME: fir.target_cpu = "x86-64"
! BOTH-SAME: fir.target_features = #llvm.target_features<["+sse"]>
