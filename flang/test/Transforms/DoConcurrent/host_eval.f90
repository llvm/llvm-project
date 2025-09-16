! REQUIRES: amdgpu-registered-target

! Tests `host_eval` clause code-gen and loop nest bounds on host vs. device.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa   \
! RUN:   -fdo-concurrent-to-openmp=device %s -o -                           \
! RUN: | FileCheck %s --check-prefix=HOST -vv

! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp            \
! RUN:   -fopenmp-is-target-device -fdo-concurrent-to-openmp=device %s -o - \
! RUN: | FileCheck %s --check-prefix=DEVICE

program do_concurrent_host_eval
    implicit none
    integer :: i, j

    do concurrent (i=1:10, j=1:20)
    end do
end program do_concurrent_host_eval

! HOST: omp.target host_eval(
! HOST-SAME:    %{{[^[:space:]]+}} -> %[[I_LB:[^,]+]],
! HOST-SAME:    %{{[^[:space:]]+}} -> %[[I_UB:[^,]+]],
! HOST-SAME:    %{{[^[:space:]]+}} -> %[[I_ST:[^,]+]],
! HOST-SAME:    %{{[^[:space:]]+}} -> %[[J_LB:[^,]+]],
! HOST-SAME:    %{{[^[:space:]]+}} -> %[[J_UB:[^,]+]],
! HOST-SAME:    %{{[^[:space:]]+}} -> %[[J_ST:[^,]+]] : {{.*}}) map_entries

! HOST: omp.loop_nest ({{.*}}, {{.*}}) : index = (%[[I_LB]], %[[J_LB]]) to
! HOST-SAME:    (%[[I_UB]], %[[J_UB]]) inclusive step
! HOST-SAME:    (%[[I_ST]], %[[J_ST]])

! DEVICE: omp.target map_entries(
! DEVICE-SAME:  %{{[^[:space:]]+}} -> %[[I_LB_MAP:[^,]+]],
! DEVICE-SAME:  %{{[^[:space:]]+}} -> %[[I_UB_MAP:[^,]+]],
! DEVICE-SAME:  %{{[^[:space:]]+}} -> %[[I_ST_MAP:[^,]+]],

! DEVICE-SAME:  %{{[^[:space:]]+}} -> %[[J_LB_MAP:[^,]+]],
! DEVICE-SAME:  %{{[^[:space:]]+}} -> %[[J_UB_MAP:[^,]+]],
! DEVICE-SAME:  %{{[^[:space:]]+}} -> %[[J_ST_MAP:[^,]+]],

! DEVICE-SAME:  %{{[^[:space:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:  %{{[^[:space:]]+}} -> %{{[^,]+}} : {{.*}})

! DEVICE: %[[I_LB_DECL:.*]]:2 = hlfir.declare %[[I_LB_MAP]]
! DEVICE: %[[I_LB:.*]] = fir.load %[[I_LB_DECL]]#1 : !fir.ref<index>

! DEVICE: %[[I_UB_DECL:.*]]:2 = hlfir.declare %[[I_UB_MAP]]
! DEVICE: %[[I_UB:.*]] = fir.load %[[I_UB_DECL]]#1 : !fir.ref<index>

! DEVICE: %[[I_ST_DECL:.*]]:2 = hlfir.declare %[[I_ST_MAP]]
! DEVICE: %[[I_ST:.*]] = fir.load %[[I_ST_DECL]]#1 : !fir.ref<index>

! DEVICE: %[[J_LB_DECL:.*]]:2 = hlfir.declare %[[J_LB_MAP]]
! DEVICE: %[[J_LB:.*]] = fir.load %[[J_LB_DECL]]#1 : !fir.ref<index>

! DEVICE: %[[J_UB_DECL:.*]]:2 = hlfir.declare %[[J_UB_MAP]]
! DEVICE: %[[J_UB:.*]] = fir.load %[[J_UB_DECL]]#1 : !fir.ref<index>

! DEVICE: %[[J_ST_DECL:.*]]:2 = hlfir.declare %[[J_ST_MAP]]
! DEVICE: %[[J_ST:.*]] = fir.load %[[J_ST_DECL]]#1 : !fir.ref<index>

! DEVICE: omp.loop_nest ({{.*}}, {{.*}}) : index = (%[[I_LB]], %[[J_LB]]) to
! DEVICE-SAME:    (%[[I_UB]], %[[J_UB]]) inclusive step
! DEVICE-SAME:    (%[[I_ST]], %[[J_ST]])
