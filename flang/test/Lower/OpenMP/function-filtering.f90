! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-HOST,LLVM-ALL %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck --check-prefixes=AIIR-HOST,AIIR-ALL %s
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-DEVICE,LLVM-ALL %s %}
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=AIIR-DEVICE,AIIR-ALL %s %}
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck --check-prefixes=AIIR-HOST,AIIR-ALL %s
! RUN: %if amdgpu-registered-target %{ bbc -target amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=AIIR-DEVICE,AIIR-ALL %s %}

! Check that the correct LLVM IR functions are kept for the host and device
! after running the whole set of translation and transformation passes from
! Fortran.

! AIIR-ALL: func.func @{{.*}}device_fn(
! AIIR-ALL: return

! LLVM-ALL: define {{.*}} @{{.*}}device_fn{{.*}}(
function device_fn() result(x)
  !$omp declare target to(device_fn) device_type(nohost)
  integer :: x
  x = 10
end function device_fn

! AIIR-ALL: func.func @{{.*}}device_fn_enter(
! AIIR-ALL: return

! LLVM-ALL: define {{.*}} @{{.*}}device_fn_enter{{.*}}(
function device_fn_enter() result(x)
  !$omp declare target enter(device_fn_enter) device_type(nohost)
  integer :: x
  x = 10
end function device_fn_enter

! AIIR-HOST: func.func @{{.*}}host_fn(
! AIIR-HOST: return
! AIIR-DEVICE-NOT: func.func {{.*}}host_fn(

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

! AIIR-ALL: func.func @{{.*}}target_subr(
! AIIR-ALL: return

! LLVM-HOST: define {{.*}} @{{.*}}target_subr{{.*}}(
! LLVM-ALL: define {{.*}} @__omp_offloading_{{.*}}_{{.*}}_target_subr__{{.*}}(
subroutine target_subr(x)
  integer, intent(out) :: x
  !$omp target map(from:x)
    x = 10
  !$omp end target
end subroutine target_subr
