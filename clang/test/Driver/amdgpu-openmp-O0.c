// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -fno-openmp-target-new-runtime -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -O0 %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "c"{{.*}}
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "ir"{{.*}}
// CHECK-NOT: -O1
