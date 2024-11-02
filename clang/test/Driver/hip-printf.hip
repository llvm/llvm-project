// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### --target=x86_64-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1"
// CHECK-NOT: "-Werror=format-nonliteral"
