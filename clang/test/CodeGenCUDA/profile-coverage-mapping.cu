// RUN: echo "GPU binary would be here" > %t
// RUN: %clang_cc1 -fprofile-instrument=clang -triple x86_64-linux-gnu -target-sdk-version=8.0 -fcuda-include-gpubinary %t -emit-llvm -o - %s | FileCheck --check-prefix=PGOGEN %s
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -triple x86_64-linux-gnu -target-sdk-version=8.0 -fcuda-include-gpubinary %t -emit-llvm -o - %s | FileCheck --check-prefix=COVMAP %s
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -triple x86_64-linux-gnu -target-sdk-version=8.0 -fcuda-include-gpubinary %t -emit-llvm-only -o - %s | FileCheck --check-prefix=MAPPING %s

#include "Inputs/cuda.h"

// PGOGEN-NOT: @__profn_{{.*kernel.*}} =
// COVMAP-COUNT-2: section "__llvm_covfun", comdat
// COVMAP-NOT: section "__llvm_covfun", comdat
// MAPPING-NOT: {{.*dfn.*}}:
// MAPPING-NOT: {{.*kernel.*}}:

__device__ void dfn(int i) {}

__global__ void kernel(int i) { dfn(i); }

void host(void) {
  kernel<<<1, 1>>>(1);
}
