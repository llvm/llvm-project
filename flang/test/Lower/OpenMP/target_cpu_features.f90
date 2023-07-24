!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -emit-fir -triple amdgcn-amd-amdhsa -target-cpu gfx908 -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

!===============================================================================
! Target_Enter Simple
!===============================================================================

!CHECK: omp.target = #omp.target<target_cpu = "gfx908",
!CHECK-SAME: target_features = "+dot3-insts,+dot4-insts,+s-memtime-inst,
!CHECK-SAME: +16-bit-insts,+s-memrealtime,+dot6-insts,+dl-insts,+image-insts,+wavefrontsize64,
!CHECK-SAME: +gfx9-insts,+gfx8-insts,+ci-insts,+dot10-insts,+dot7-insts,
!CHECK-SAME: +dot1-insts,+dot5-insts,+mai-insts,+dpp,+dot2-insts">
!CHECK-LABEL: func.func @_QPomp_target_simple()
subroutine omp_target_simple
  ! Directive needed to prevent subroutine from being filtered out when
  ! compiling for the device.
  !$omp declare target
end subroutine omp_target_simple

