! RUN: %flang_fc1 -fopenmp -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-HOST %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-DEVICE %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s

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
! MLIR-ALL: func.func private @_QFhost_parent_procedurePtarget_internal_proc(

! LLVM-HOST: define {{.*}} @_QFhost_parent_procedurePtarget_internal_proc(
! LLVM-HOST: define {{.*}} @__omp_offloading_{{.*}}QFhost_parent_procedurePtarget_internal_proc{{.*}}(

! FIXME: the last check above should also work on the host, but the offload function
! is deleted because it is private and all its usages have been removed in the
! device code. Maybe the private attribute should be removed on internal
! functions while filtering?
subroutine target_internal_proc(x)
  integer, intent(out) :: x
  !$omp target map(from:x)
    x = 10
  !$omp end target
end subroutine
end subroutine
