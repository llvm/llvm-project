// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -c -target x86_64-linux-gnu -fgpu-rdc \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck %s

// CHECK: {{.*}}clang-offload-bundler{{.*}}"-outputs=hip-output-file-name.o"
