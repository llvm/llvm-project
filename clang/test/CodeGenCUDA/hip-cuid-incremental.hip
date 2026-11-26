// Check that the __hip_cuid_ global is emitted in normal HIP compilation, but
// skipped under incremental extensions (clang-repl), where it would otherwise
// cause duplicate-symbol errors at JIT link time. This applies to both host and
// device code generation.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -emit-llvm -cuid=abcd -o - %s | FileCheck --check-prefix=NORMAL %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -emit-llvm -fincremental-extensions -cuid=abcd -o - %s | FileCheck --check-prefix=INCR %s

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fcuda-is-device -x hip -emit-llvm -cuid=abcd -o - %s | FileCheck --check-prefix=DEV-NORMAL %s

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fcuda-is-device -x hip -emit-llvm -fincremental-extensions -cuid=abcd -o - %s | FileCheck --check-prefix=DEV-INCR %s

#include "Inputs/cuda.h"

__global__ void kernel() {}

// NORMAL: @__hip_cuid_
// INCR-NOT: @__hip_cuid_

// DEV-NORMAL: @__hip_cuid_
// DEV-INCR-NOT: @__hip_cuid_
