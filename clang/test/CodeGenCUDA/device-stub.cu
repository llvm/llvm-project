// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

#include "../SemaCUDA/cuda.h"

// Test that we build the correct number of calls to cudaSetupArgument followed
// by a call to cudaLaunch.

// CHECK: define{{.*}}kernelfunc
// CHECK: call{{.*}}cudaSetupArgument
// CHECK: call{{.*}}cudaSetupArgument
// CHECK: call{{.*}}cudaSetupArgument
// CHECK: call{{.*}}cudaLaunch
__global__ void kernelfunc(int i, int j, int k) {}
