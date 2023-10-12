! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s
! RUN: %flang_fc1 -fopenmp -emit-mlir %s -o - | FileCheck --check-prefix=MLIR %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-mlir %s -o - | FileCheck --check-prefix=MLIR %s
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: bbc -fopenmp -fopenmp-is-target-device -emit-fir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s

! MLIR: func.func @{{.*}}implicit_invocation() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>}
! MLIR: return
! LLVM: define {{.*}} @{{.*}}implicit_invocation{{.*}}(
subroutine implicit_invocation()
end subroutine implicit_invocation

! MLIR: func.func @{{.*}}declaretarget() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>}
! MLIR: return
! LLVM: define {{.*}} @{{.*}}declaretarget{{.*}}(
subroutine declaretarget()
!$omp declare target to(declaretarget) device_type(nohost)
    call implicit_invocation()
end subroutine declaretarget

! MLIR: func.func @{{.*}}no_declaretarget() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>}
! MLIR: return
! LLVM: define {{.*}} @{{.*}}no_declaretarget{{.*}}(
subroutine no_declaretarget()
end subroutine no_declaretarget

! MLIR-HOST: func.func @{{.*}}main(
! MLIR-HOST-NOT: func.func @{{.*}}main_omp_outline{{.*}}()
! MLIR-DEVICE-NOT: func.func @{{.*}}main(
! MLIR-DEVICE: func.func @{{.*}}main_omp_outline{{.*}}() attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QQmain"}
! MLIR-ALL: return

! LLVM-HOST: define {{.*}} @{{.*}}main{{.*}}(
! LLVM-HOST-NOT: {{.*}} @{{.*}}__omp_offloading{{.*}}main_{{.*}}(
! LLVM-DEVICE-NOT: {{.*}} @{{.*}}main{{.*}}(
! LLVM-DEVICE: define {{.*}} @{{.*}}__omp_offloading{{.*}}main_{{.*}}(
program main
!$omp target
    call declaretarget()
    call no_declaretarget()
!$omp end target
end program main
