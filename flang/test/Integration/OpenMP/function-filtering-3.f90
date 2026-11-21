!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-ALL,LLVM-HOST %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | tco -test-gen | FileCheck --check-prefixes=MLIR-ALL,MLIR-HOST %s
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-ALL,LLVM-DEVICE %s %}
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | tco -test-gen | FileCheck --check-prefixes=MLIR-ALL,MLIR-DEVICE %s %}
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | tco -test-gen | FileCheck --check-prefixes=MLIR-ALL,MLIR-HOST %s
! RUN: %if amdgpu-registered-target %{ bbc -target amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | tco -test-gen | FileCheck --check-prefixes=MLIR-ALL,MLIR-DEVICE %s %}

! Check that the correct LLVM IR functions are kept for the host and device
! after running the whole set of translation and transformation passes from
! Fortran.

! MLIR-HOST: llvm.func @{{.*}}host_parent_procedure(
! MLIR-HOST: llvm.return
! MLIR-DEVICE-NOT: llvm.func {{.*}}host_parent_procedure(

! LLVM-HOST: define {{.*}} @host_parent_procedure{{.*}}(
! LLVM-DEVICE-NOT: {{.*}} @{{.*}}_host_parent_procedure{{.*}}(
subroutine host_parent_procedure(x)
  integer, intent(out) :: x
  call target_internal_proc(x)
contains

! MLIR-ALL: llvm.func {{.*}}@_QFhost_parent_procedurePtarget_internal_proc(
! MLIR-ALL: llvm.func {{.*}}@_QFhost_parent_procedurePdeclare_target_internal_proc

! LLVM-HOST: define {{.*}} @_QFhost_parent_procedurePtarget_internal_proc(
! LLVM-DEVICE-NOT: define {{.*}} @_QFhost_parent_procedurePtarget_internal_proc(
! LLVM-ALL: define {{.*}} void @_QFhost_parent_procedurePdeclare_target_internal_proc

! LLVM-ALL: define {{.*}} @__omp_offloading_{{.*}}QFhost_parent_procedurePtarget_internal_proc{{.*}}(

subroutine target_internal_proc(x)
  integer, intent(out) :: x
  !$omp target map(from:x)
    x = 10
    call declare_target_internal_proc()
  !$omp end target
end subroutine

subroutine declare_target_internal_proc()
  !$omp declare target enter(declare_target_internal_proc) device_type(nohost)
end subroutine
end subroutine
