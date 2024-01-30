! REQUIRES: amdgpu-registered-target
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefixes=ALL,NONE
! RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefixes=ALL,TRIPLE
! RUN: %flang_fc1 -emit-fir -target-cpu gfx90a %s -o - | FileCheck %s --check-prefixes=ALL,CPU
! RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa -target-cpu gfx90a %s -o - | FileCheck %s --check-prefixes=ALL,BOTH

! ALL: module attributes {

! NONE-NOT: fir.target_cpu
! NONE-NOT: fir.target_features

! TRIPLE-SAME: fir.target_cpu = "generic-hsa"
! TRIPLE-NOT: fir.target_features

! CPU-SAME: fir.target_cpu = "gfx90a"
! CPU-NOT: fir.target_features

! BOTH-SAME: fir.target_cpu = "gfx90a"
! BOTH-SAME: fir.target_features = #llvm.target_features<[
! BOTH-SAME: "+gfx90a-insts"
! BOTH-SAME: ]>
