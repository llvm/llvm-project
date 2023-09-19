! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-HOST,LLVM-ALL %s
! RUN: %flang_fc1 -fopenmp -emit-mlir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-DEVICE,LLVM-ALL %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-mlir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-fir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s

! Check that the correct LLVM IR functions are kept for the host and device
! after running the whole set of translation and transformation passes from
! Fortran.

! MLIR-ALL: func.func @{{.*}}device_fn(
! MLIR-ALL: return

! LLVM-ALL: define {{.*}} @{{.*}}device_fn{{.*}}(
function device_fn() result(x)
  !$omp declare target to(device_fn) device_type(nohost)
  integer :: x
  x = 10
end function device_fn

! MLIR-HOST: func.func @{{.*}}host_fn(
! MLIR-HOST: return
! MLIR-DEVICE: func.func private @{{.*}}host_fn(
! MLIR-DEVICE-NOT: return

! LLVM-HOST: define {{.*}} @{{.*}}host_fn{{.*}}(
! LLVM-DEVICE-NOT: {{.*}} @{{.*}}host_fn{{.*}}(
function host_fn() result(x)
  !$omp declare target to(host_fn) device_type(host)
  integer :: x
  x = 10
end function host_fn

! MLIR-HOST: func.func @{{.*}}target_subr(
! MLIR-HOST: return
! MLIR-HOST-NOT: func.func @{{.*}}target_subr_omp_outline_0(
! MLIR-DEVICE-NOT: func.func @{{.*}}target_subr(
! MLIR-DEVICE: func.func @{{.*}}target_subr_omp_outline_0(
! MLIR-DEVICE: return

! LLVM-ALL-NOT: define {{.*}} @{{.*}}target_subr_omp_outline_0{{.*}}(
! LLVM-HOST: define {{.*}} @{{.*}}target_subr{{.*}}(
! LLVM-DEVICE-NOT: {{.*}} @{{.*}}target_subr{{.*}}(
! LLVM-ALL: define {{.*}} @__omp_offloading_{{.*}}_{{.*}}_target_subr__{{.*}}(
subroutine target_subr(x)
  integer, intent(out) :: x
  !$omp target map(from:x)
    x = 10
  !$omp end target
end subroutine target_subr
