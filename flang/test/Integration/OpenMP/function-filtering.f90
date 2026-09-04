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

! MLIR-ALL: llvm.func @{{.*}}device_fn(
! MLIR-ALL: llvm.return

! LLVM-ALL: define {{.*}} @{{.*}}device_fn{{.*}}(
function device_fn() result(x)
  !$omp declare target to(device_fn) device_type(nohost)
  integer :: x
  x = 10
end function device_fn

! MLIR-ALL: llvm.func @{{.*}}device_fn_enter(
! MLIR-ALL: llvm.return

! LLVM-ALL: define {{.*}} @{{.*}}device_fn_enter{{.*}}(
function device_fn_enter() result(x)
  !$omp declare target enter(device_fn_enter) device_type(nohost)
  integer :: x
  x = 10
end function device_fn_enter

! MLIR-HOST: llvm.func @{{.*}}host_fn(
! MLIR-HOST: llvm.return
! MLIR-DEVICE-NOT: llvm.func {{.*}}host_fn(

! LLVM-HOST: define {{.*}} @{{.*}}host_fn{{.*}}(
! LLVM-DEVICE-NOT: {{.*}} @{{.*}}host_fn{{.*}}(
function host_fn() result(x)
  !$omp declare target to(host_fn) device_type(host)
  integer :: x
  x = 10
end function host_fn

! LLVM-HOST: define {{.*}} @{{.*}}host_fn_enter{{.*}}(
! LLVM-DEVICE-NOT: {{.*}} @{{.*}}host_fn_enter{{.*}}(
function host_fn_enter() result(x)
  !$omp declare target enter(host_fn_enter) device_type(host)
  integer :: x
  x = 10
end function host_fn_enter

! MLIR-ALL: llvm.func @{{.*}}target_subr(
! MLIR-ALL: llvm.return

! LLVM-HOST: define {{.*}} @{{.*}}target_subr{{.*}}(
! LLVM-ALL: define {{.*}} @__omp_offloading_{{.*}}_{{.*}}_target_subr__{{.*}}(
subroutine target_subr(x)
  integer, intent(out) :: x
  !$omp target map(from:x)
    x = 10
  !$omp end target
end subroutine target_subr
