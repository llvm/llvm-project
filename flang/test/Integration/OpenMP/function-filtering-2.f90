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

program main
    ! MLIR-ALL: llvm.func @{{.+}}main(
    ! MLIR-ALL: omp.target
    ! MLIR-ALL: llvm.return
    !$omp target
        call declaretarget()
        call declaretarget_enter()
        call no_declaretarget()
    !$omp end target

    contains
    ! MLIR-ALL: llvm.func{{.*}} @{{.*}}implicit_invocation() attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), implicit = true>{{.*}}}
    ! MLIR-ALL: llvm.return
    ! LLVM-ALL: define {{.*}} @{{.*}}implicit_invocation{{.*}}(
    subroutine implicit_invocation()
    end subroutine implicit_invocation

    ! MLIR-ALL: llvm.func{{.*}} @{{.*}}declaretarget() attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>{{.*}}}
    ! MLIR-ALL: llvm.return
    ! LLVM-ALL: define {{.*}} @{{.*}}declaretarget{{.*}}(
    subroutine declaretarget()
    !$omp declare target to(declaretarget) device_type(nohost)
        call implicit_invocation()
    end subroutine declaretarget

    ! MLIR-ALL: llvm.func{{.*}} @{{.*}}declaretarget_enter() attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>{{.*}}}
    ! MLIR-ALL: llvm.return
    ! LLVM-ALL: define {{.*}} @{{.*}}declaretarget_enter{{.*}}(
    subroutine declaretarget_enter()
    !$omp declare target enter(declaretarget_enter) device_type(nohost)
        call implicit_invocation()
    end subroutine declaretarget_enter

    ! MLIR-ALL: llvm.func{{.*}} @{{.*}}no_declaretarget() attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), implicit = true>{{.*}}}
    ! MLIR-ALL: llvm.return
    ! LLVM-ALL: define {{.*}} @{{.*}}no_declaretarget{{.*}}(
    subroutine no_declaretarget()
    end subroutine no_declaretarget

    ! MLIR-HOST: llvm.func{{.*}} @main(
    ! MLIR-DEVICE-NOT: llvm.func{{.*}} @main(
    ! MLIR-HOST: llvm.return

    ! LLVM-HOST: define {{.*}} @{{.*}}main{{.*}}(
    ! LLVM-HOST: {{.*}} @{{.*}}__omp_offloading{{.*}}main_{{.*}}(
    ! LLVM-DEVICE-NOT: {{.*}} @{{.*}}main{{.*}}(
    ! LLVM-DEVICE: define {{.*}} @{{.*}}__omp_offloading{{.*}}main_{{.*}}(
end program main
