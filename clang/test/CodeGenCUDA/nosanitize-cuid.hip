// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fsanitize=address -fcuda-is-device \
// RUN:    -emit-llvm -cuid=abcd -o - %s | FileCheck  %s

#include "Inputs/cuda.h"

// CHECK: @__hip_cuid_{{.*}} = {{.*}} no_sanitize_address
