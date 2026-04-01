! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM,LLVM-HOST %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck --check-prefix=AIIR %s
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM,LLVM-DEVICE %s %}
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefix=AIIR %s %}
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck --check-prefixes=AIIR-HOST,AIIR-ALL %s
! RUN: %if amdgpu-registered-target %{ bbc -target amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=AIIR-DEVICE,AIIR-ALL %s %}

! AIIR: func.func @{{.*}}implicit_invocation() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>}
! AIIR: return
! LLVM: define {{.*}} @{{.*}}implicit_invocation{{.*}}(
subroutine implicit_invocation()
end subroutine implicit_invocation

! AIIR: func.func @{{.*}}declaretarget() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>}
! AIIR: return
! LLVM: define {{.*}} @{{.*}}declaretarget{{.*}}(
subroutine declaretarget()
!$omp declare target to(declaretarget) device_type(nohost)
    call implicit_invocation()
end subroutine declaretarget

! AIIR: func.func @{{.*}}declaretarget_enter() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter), automap = false>}
! AIIR: return
! LLVM: define {{.*}} @{{.*}}declaretarget_enter{{.*}}(
subroutine declaretarget_enter()
!$omp declare target enter(declaretarget_enter) device_type(nohost)
    call implicit_invocation()
end subroutine declaretarget_enter

! AIIR: func.func @{{.*}}no_declaretarget() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>}
! AIIR: return
! LLVM: define {{.*}} @{{.*}}no_declaretarget{{.*}}(
subroutine no_declaretarget()
end subroutine no_declaretarget

! AIIR-HOST: func.func @{{.*}}main(
! AIIR-DEVICE-NOT: func.func @{{.*}}main(
! AIIR-ALL: return

! LLVM-HOST: define {{.*}} @{{.*}}main{{.*}}(
! LLVM-HOST: {{.*}} @{{.*}}__omp_offloading{{.*}}main_{{.*}}(
! LLVM-DEVICE-NOT: {{.*}} @{{.*}}main{{.*}}(
! LLVM-DEVICE: define {{.*}} @{{.*}}__omp_offloading{{.*}}main_{{.*}}(
program main
!$omp target
    call declaretarget()
    call no_declaretarget()
!$omp end target
end program main
