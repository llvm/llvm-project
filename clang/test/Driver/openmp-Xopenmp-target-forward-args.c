// Check that -Xopenmp-target forwards arbitrary arguments, not just -march

// RUN:   %clang -### --target=x86_64-pc-linux -no-canonical-prefixes -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900 -Xopenmp-target=amdgcn-amd-amdhsa -ffinite-math-only -nogpulib %s 2>&1 \
// RUN:   | FileCheck %s

// Flag should only apply to device, not the host.

// CHECK-NOT: -ffinite-math-only
// CHECK: "-triple" "amdgcn-amd-amdhsa" {{.*}} "-ffinite-math-only"
// CHECK-NOT: -ffinite-math-only
