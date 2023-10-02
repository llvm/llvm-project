!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -emit-hlfir -triple amdgcn-amd-amdhsa -target-cpu gfx908 -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

!===============================================================================
! Target_Enter Simple
!===============================================================================

!CHECK: omp.target = #omp.target<target_cpu = "gfx908",
!CHECK-SAME: target_features = "+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,
!CHECK-SAME: +dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,
!CHECK-SAME: +gfx8-insts,+gfx9-insts,+gws,+image-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,
!CHECK-SAME: +wavefrontsize64">
!CHECK-LABEL: func.func @_QPomp_target_simple()
subroutine omp_target_simple
  ! Directive needed to prevent subroutine from being filtered out when
  ! compiling for the device.
  !$omp declare target
end subroutine omp_target_simple

