!REQUIRES: amdgpu-registered-target, nvptx-registered-target
!RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa -target-cpu gfx908 -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -triple nvptx64-nvidia-cuda -target-cpu sm_80 -fopenmp -fopenmp-is-target-device %s -o - | FileCheck --check-prefix=NVPTX %s


!===============================================================================
! Target_Enter Simple
!===============================================================================

!CHECK: omp.target = #omp.target<target_cpu = "gfx908",
!CHECK-SAME: target_features = "+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,
!CHECK-SAME: +dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,
!CHECK-SAME: +gfx8-insts,+gfx9-insts,+gws,+image-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,
!CHECK-SAME: +wavefrontsize64">
!NVPTX: omp.target = #omp.target<target_cpu = "sm_80", target_features = "+ptx61,+sm_80">
!CHECK-LABEL: func.func @_QPomp_target_simple()
subroutine omp_target_simple
  ! Directive needed to prevent subroutine from being filtered out when
  ! compiling for the device.
  !$omp declare target
end subroutine omp_target_simple

