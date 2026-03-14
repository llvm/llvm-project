// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix DEVICE

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-pc-linux-gnu -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-pc-linux-gnu -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-i386-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-i386-host.bc -o - | FileCheck %s --check-prefix DEVICE

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm-bc %s -o %t-x86_64-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86_64-host.bc -o - | FileCheck %s --check-prefix DEVICE

// expected-no-diagnostics

#pragma omp declare target
#pragma omp begin declare variant match(device = {kind(nohost)})
int G1;
static int G2;
#pragma omp end declare variant
#pragma omp end declare target

#pragma omp begin declare target device_type(nohost)
int G3;
static int G4;
#pragma omp end declare target

#pragma omp declare target
int G5;
static int G6;
#pragma omp end declare target

#pragma omp declare target to(G5, G6) device_type(nohost)

#pragma omp begin declare target device_type(host)
int G7;
static int G8;
#pragma omp end declare target

#pragma omp declare target
int G9;
static int G10;
#pragma omp end declare target

int G11;
static int G12;
#pragma omp declare target to(G9, G10, G11, G12) device_type(host)

// TODO: The code below should probably work but it is not 100% clear.
#if 0
#pragma omp declare target
#pragma omp begin declare variant match(device = {kind(host)})
int GX;
static int GY;
#pragma omp end declare variant
#pragma omp end declare target
#endif

// TODO: It is odd, probably wrong, that we don't mangle all variables.

// DEVICE-DAG: @G1 = {{.*}}global i32 0, align 4
// DEVICE-DAG: @_ZL2G2 = internal {{.*}}global i32 0, align 4
// DEVICE-DAG: @G3 = {{.*}}global i32 0, align 4
// DEVICE-DAG: @_ZL2G4 = internal {{.*}}global i32 0, align 4
// DEVICE-DAG: @G5 = {{.*}}global i32 0, align 4
// DEVICE-DAG: @_ZL2G6 = internal {{.*}}global i32 0, align 4
// DEVICE-NOT: ref
// DEVICE-NOT: llvm.used
// DEVICE-NOT: omp_offload

// HOST-DAG: @G7 = global i32 0, align 4
// HOST-DAG: @_ZL2G8 = internal global i32 0, align 4
// HOST-DAG: @G9 = global i32 0, align 4
// HOST-DAG: @_ZL3G10 = internal global i32 0, align 4
// HOST-DAG: @G11 = global i32 0, align 4
// HOST-DAG: @_ZL3G12 = internal global i32 0, align 4
