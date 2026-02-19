! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-HOST,LLVM-ALL %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-ALL %s
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-DEVICE,LLVM-ALL %s %}
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefix=MLIR-DEVICE,MLIR-ALL %s %}
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-ALL %s
! RUN: %if amdgpu-registered-target %{ bbc -target amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s %}

! MLIR-ALL: func.func @{{.*}}implicit_invocation() attributes {
! MLIR-DEVICE: llvm.linkage = #llvm.linkage<internal>
! MLIR-ALL-SAME: omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>}
! MLIR-ALL: return
! LLVM-ALL: define {{.*}} @{{.*}}implicit_invocation{{.*}}(
subroutine implicit_invocation()
end subroutine implicit_invocation

! MLIR-ALL: func.func @{{.*}}declaretarget() attributes {
! MLIR-DEVICE: llvm.linkage = #llvm.linkage<internal>
! MLIR-ALL-SAME: omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>}
! MLIR-ALL: return
! LLVM-ALL: define {{.*}} @{{.*}}declaretarget{{.*}}(
subroutine declaretarget()
!$omp declare target to(declaretarget) device_type(nohost)
    call implicit_invocation()
end subroutine declaretarget

! MLIR-ALL: func.func @{{.*}}declaretarget_enter() attributes {
! MLIR-DEVICE: llvm.linkage = #llvm.linkage<internal>
! MLIR-ALL-SAME: omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter), automap = false>}
! MLIR-ALL: return
! LLVM-ALL: define {{.*}} @{{.*}}declaretarget_enter{{.*}}(
subroutine declaretarget_enter()
!$omp declare target enter(declaretarget_enter) device_type(nohost)
    call implicit_invocation()
end subroutine declaretarget_enter

! MLIR-ALL: func.func @{{.*}}no_declaretarget() attributes {
! MLIR-DEVICE: llvm.linkage = #llvm.linkage<internal>
! MLIR-ALL-SAME: omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>}
! MLIR-ALL: return
! LLVM-ALL: define {{.*}} @{{.*}}no_declaretarget{{.*}}(
subroutine no_declaretarget()
end subroutine no_declaretarget

! MLIR-ALL: func.func @{{.*}}main(
! MLIR-ALL: return

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
