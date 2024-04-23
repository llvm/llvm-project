!REQUIRES: amdgpu-registered-target, nvptx-registered-target
!RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa -target-cpu gfx908 -fopenmp -fopenmp-is-target-device %s -o - | FileCheck --check-prefix=AMDGCN %s
!RUN: %flang_fc1 -emit-fir -triple nvptx64-nvidia-cuda -target-cpu sm_80 -fopenmp -fopenmp-is-target-device %s -o - | FileCheck --check-prefix=NVPTX %s

!===============================================================================
! Target_Enter Simple
!===============================================================================

!AMDGCN: module attributes {
!AMDGCN-SAME: fir.target_cpu = "gfx908"
!AMDGCN-SAME: fir.target_features = #llvm.target_features<["+16-bit-insts", "+ci-insts",
!AMDGCN-SAME: "+dl-insts", "+dot1-insts", "+dot10-insts", "+dot2-insts", "+dot3-insts",
!AMDGCN-SAME: "+dot4-insts", "+dot5-insts", "+dot6-insts", "+dot7-insts", "+dpp",
!AMDGCN-SAME: "+gfx8-insts", "+gfx9-insts", "+gws", "+image-insts", "+mai-insts",
!AMDGCN-SAME: "+s-memrealtime", "+s-memtime-inst", "+wavefrontsize64"]>

!NVPTX: module attributes {
!NVPTX-SAME: fir.target_cpu = "sm_80"
!NVPTX-SAME: fir.target_features = #llvm.target_features<["+ptx61", "+sm_80"]>
