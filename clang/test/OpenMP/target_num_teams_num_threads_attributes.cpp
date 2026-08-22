// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgpu-amd-amdhsa -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s 
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgpu9.00-amd-amdhsa -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64 -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple nvptx64 -fopenmp-targets=nvptx64 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s 
// RUN: %clang_cc1 -target-cpu sm_80 -fopenmp -x c++ -std=c++11 -triple nvptx64 -fopenmp-targets=nvptx64 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=spirv64 -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple spirv64 -fopenmp-targets=spirv64 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s 

// expected-no-diagnostics


#ifndef HEADER
#define HEADER


void default_val_num_teams() {
    #pragma omp target simd
    for (int i = 0; i < 22; i++)
        int a_var;
}

void foo1() {
    #pragma omp target teams num_teams(22)
    { int a_var; }
}

void foo2() {
    #pragma omp target teams distribute num_teams(22)
    for (int i = 0; i < 22; i++)
        int a_var;
}

void foo3() {
    #pragma omp target teams distribute parallel for num_teams(22)
    for (int i = 0; i < 22; i++)
        int a_var;
}

void bar1() {
    #pragma omp target teams num_teams(22)
    { int a_var; }
}

void bar2() {
    #pragma omp target teams distribute num_teams(33)
    for (int i = 0; i < 22; i++)
        int a_var;
}

void bar3() {
    #pragma omp target teams distribute parallel for num_teams(44)
    for (int i = 0; i < 22; i++)
        int a_var;
}

void const_int() {
    const int NT = 22;
    #pragma omp target teams num_teams(NT)
    { int a_var; }
}

void thread_limit() {
    #pragma omp target teams thread_limit(22)
    { int a_var; }
}

void num_threads() {
    #pragma omp target teams distribute parallel for thread_limit(22) num_threads(11)
    for (int i = 0; i < 22; i++)
        int a_var;
}

void threads_and_teams() {
    #pragma omp target teams distribute parallel for thread_limit(22) num_teams(33)
    for (int i = 0; i < 22; i++)
        int a_var;
}

void thread_limit_at_max() {
    #pragma omp target teams thread_limit(1024)
    { int a_var; }
}

#endif


// CHECK:      "omp_target_num_teams"="1"
// CHECK:      "omp_target_num_teams"="22"
// CHECK:      "omp_target_num_teams"="33"
// CHECK:      "omp_target_num_teams"="44"

// A generic mode kernel reserves one warp for the main thread, so its bound is
// the thread_limit clause plus the warp size, which differs across the targets
// this test runs on. The kernels below are SPMD and keep the value the program
// asked for.
// CHECK:      "omp_target_thread_limit"="{{54|86}}"

// CHECK:      "omp_target_thread_limit"="11"

// CHECK:      "omp_target_num_teams"="33"
// CHECK-SAME: "omp_target_thread_limit"="22"

// A thread_limit already at the target's maximum has no room for the extra
// warp, so the last one goes to the main thread instead of growing the block.
// Every target this test runs on allows 1024, whatever its warp size is.
// CHECK:      "omp_target_thread_limit"="1024"
