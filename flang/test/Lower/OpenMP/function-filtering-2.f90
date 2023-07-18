! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-HOST %s
! RUN: %flang_fc1 -fopenmp -emit-mlir %s -o - | FileCheck --check-prefix=MLIR-HOST %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-DEVICE %s
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-mlir %s -o - | FileCheck --check-prefix=MLIR-DEVICE %s

! MLIR-HOST: func.func @{{.*}}implicit_invocation(
! MLIR-DEVICE: func.func @{{.*}}implicit_invocation() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>}
! LLVM-HOST: define {{.*}} @{{.*}}implicit_invocation{{.*}}(
! LLVM-DEVICE: define {{.*}} @{{.*}}implicit_invocation{{.*}}(
subroutine implicit_invocation()
end subroutine implicit_invocation

! MLIR-HOST: func.func @{{.*}}declaretarget(
! MLIR-DEVICE: func.func @{{.*}}declaretarget() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>}
! LLVM-HOST: define {{.*}} @{{.*}}declaretarget{{.*}}(
! LLVM-DEVICE: define {{.*}} @{{.*}}declaretarget{{.*}}(
subroutine declaretarget()
!$omp declare target to(declaretarget) device_type(nohost)
    call implicit_invocation()
end subroutine declaretarget

! MLIR-HOST: func.func @{{.*}}no_declaretarget(
! MLIR-DEVICE: func.func @{{.*}}no_declaretarget() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>}
! LLVM-HOST: define {{.*}} @{{.*}}no_declaretarget{{.*}}(
! LLVM-DEVICE: define {{.*}} @{{.*}}no_declaretarget{{.*}}(
subroutine no_declaretarget()
end subroutine no_declaretarget

! MLIR-HOST: func.func @{{.*}}main(
! MLIR-DEVICE: func.func @{{.*}}main_omp_outline{{.*}}() attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QQmain"} 
! LLVM-HOST: define {{.*}} @{{.*}}main{{.*}}(
! LLVM-DEVICE: define {{.*}} @{{.*}}__omp_offloading{{.*}}main_{{.*}}(
program main
!$omp target
    call declaretarget()
    call no_declaretarget()
!$omp end target
end program main
