! RUN: %flang_fc1 -fopenmp -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-HOST,LLVM-ALL %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-DEVICE,LLVM-ALL %s %}
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s %}
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: %if amdgpu-registered-target %{ bbc -target amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s %}

! Check that the correct LLVM IR functions are kept for the host and device
! after running the whole set of translation and transformation passes from
! Fortran.

! MLIR-HOST: func.func @{{.*}}host_parent_procedure(
! MLIR-HOST: return
! MLIR-DEVICE-NOT: func.func {{.*}}host_parent_procedure(

! LLVM-HOST: define {{.*}} @host_parent_procedure{{.*}}(
! LLVM-DEVICE-NOT: {{.*}} @{{.*}}_host_parent_procedure{{.*}}(
subroutine host_parent_procedure(x)
  integer, intent(out) :: x
  call target_internal_proc(x)
contains
! MLIR-ALL: func.func {{.*}}@_QFhost_parent_procedurePtarget_internal_proc(

! LLVM-HOST: define {{.*}} @_QFhost_parent_procedurePtarget_internal_proc(
! LLVM-DEVICE-NOT: define {{.*}} @_QFhost_parent_procedurePtarget_internal_proc(
! LLVM-ALL: define {{.*}} @__omp_offloading_{{.*}}QFhost_parent_procedurePtarget_internal_proc{{.*}}(

subroutine target_internal_proc(x)
  integer, intent(out) :: x
  !$omp target map(from:x)
    x = 10
  !$omp end target
end subroutine
end subroutine
