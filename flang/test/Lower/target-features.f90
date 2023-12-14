! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefixes=ALL,NONE
! RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefixes=ALL,TRIPLE
! RUN: %flang_fc1 -emit-fir -target-cpu gfx90a %s -o - | FileCheck %s --check-prefixes=ALL,CPU
! RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa -target-cpu gfx90a %s -o - | FileCheck %s --check-prefixes=ALL,BOTH

! ALL-LABEL: func.func @_QPfoo()

! NONE-NOT: target_cpu
! NONE-NOT: target_features

! TRIPLE-SAME: target_cpu = "generic-hsa"
! TRIPLE-NOT: target_features

! CPU-SAME: target_cpu = "gfx90a"
! CPU-NOT: target_features

! BOTH-SAME: target_cpu = "gfx90a"
! BOTH-SAME: target_features = #llvm.target_features<[
! BOTH-SAME: "+gfx90a-insts"
! BOTH-SAME: ]>
subroutine foo
end subroutine
