! REQUIRES: amdgpu-registered-target
! RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa -target-cpu gfx90a %s -o - | FileCheck %s --check-prefixes=ALL,CPU
! RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa -target-feature +sse %s -o - | FileCheck %s --check-prefixes=ALL,FEATURE
! RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa -target-cpu gfx90a -target-feature +sse %s -o - | FileCheck %s --check-prefixes=ALL,BOTH

! ALL: module attributes {

! CPU-SAME: fir.target_cpu = "gfx90a"
! CPU-SAME: fir.target_features = #llvm.target_features<[
! CPU-SAME: "+gfx90a-insts"
! CPU-SAME: ]>

! FEATURE-SAME: fir.target_features = #llvm.target_features<[
! FEATURE-NOT:  "+gfx90a-insts"
! FEATURE-SAME: "+sse"
! FEATURE-SAME: ]>

! BOTH-SAME: fir.target_cpu = "gfx90a"
! BOTH-SAME: fir.target_features = #llvm.target_features<[
! BOTH-SAME: "+gfx90a-insts"
! BOTH-SAME: "+sse"
! BOTH-SAME: ]>
