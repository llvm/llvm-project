// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Check that -ggdb correctly differentiates device/host code

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpulib -ggdb \
// RUN:   -x hip --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK %s

// CHECK: {{.*}}clang{{.*}}"-triple" "amdgcn-amd-amdhsa"
// CHECK-DAG: "-debug-info-kind=line-tables-only"
// CHECK-DAG: "-disable-O0-optnone"
// CHECK-DAG: "-disable-O0-noinline"
// CHECK-DAG: "-debugger-tuning=gdb"
// CHECK-LABEL: clang-offload-bundler
// CHECK: {{.*}}clang{{.*}}"-triple" "x86_64-unknown-linux-gnu"
// CHECK-NOT: "-disable-O0-optnone"
// CHECK-NOT: "-disable-O0-noinline"
// CHECK-DAG: "-debug-info-kind=limited"
// CHECK-DAG: "-debugger-tuning=gdb"
// CHECK-NOT: "-disable-O0-optnone"
// CHECK-NOT: "-disable-O0-noinline"
