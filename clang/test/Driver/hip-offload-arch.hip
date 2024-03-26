// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx1030 \
// RUN:   --offload-arch=gfx1031 \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx1031"}}
// CHECK: {{"[^"]*clang[^"]*".* "-target-cpu" "gfx1030"}}
