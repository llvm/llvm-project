! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-HOST,LLVM-ALL %s
! RUN: %flang_fc1 -fopenmp -emit-mlir %s -o - | FileCheck --check-prefix=MLIR-HOST %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-DEVICE,LLVM-ALL %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-mlir %s -o - | FileCheck --check-prefix=MLIR-DEVICE %s

! Check that the correct LLVM IR functions are kept for the host and device
! after running the whole set of translation and transformation passes from
! Fortran.

! DISABLED, this portion of the test is disabled via the removal of the colon for the time 
! being as filtering is enabled for device only for the time being while a fix is in progress. 
! MLIR-HOST-NOT func.func @{{.*}}device_fn(
! LLVM-HOST-NOT define {{.*}} @{{.*}}device_fn{{.*}}(

! MLIR-DEVICE: func.func @{{.*}}device_fn(
! LLVM-DEVICE: define {{.*}} @{{.*}}device_fn{{.*}}(
function device_fn() result(x)
  !$omp declare target to(device_fn) device_type(nohost)
  integer :: x
  x = 10
end function device_fn

! MLIR-HOST: func.func @{{.*}}host_fn(
! MLIR-DEVICE-NOT: func.func @{{.*}}host_fn(
! LLVM-HOST: define {{.*}} @{{.*}}host_fn{{.*}}(
! LLVM-DEVICE-NOT: define {{.*}} @{{.*}}host_fn{{.*}}(
function host_fn() result(x)
  !$omp declare target to(host_fn) device_type(host)
  integer :: x
  x = 10
end function host_fn

! MLIR-HOST: func.func @{{.*}}target_subr(
! MLIR-HOST-NOT: func.func @{{.*}}target_subr_omp_outline_0(
! MLIR-DEVICE-NOT: func.func @{{.*}}target_subr(
! MLIR-DEVICE: func.func @{{.*}}target_subr_omp_outline_0(

! LLVM-ALL-NOT: define {{.*}} @{{.*}}target_subr_omp_outline_0{{.*}}(
! LLVM-HOST: define {{.*}} @{{.*}}target_subr{{.*}}(
! LLVM-DEVICE-NOT: define {{.*}} @{{.*}}target_subr{{.*}}(
! LLVM-ALL: define {{.*}} @__omp_offloading_{{.*}}_{{.*}}_target_subr__{{.*}}(
subroutine target_subr(x)
  integer, intent(out) :: x
  !$omp target map(from:x)
    x = 10
  !$omp end target
end subroutine target_subr
