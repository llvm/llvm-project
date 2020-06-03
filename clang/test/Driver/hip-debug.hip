// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Check that -ggdb correctly differentiates device/host code, and that the
// -amdgpu-spill-cfi-saved-regs option is supplied during actual code-gen (i.e.
// it is in the llc command-line for the device).

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpulib -ggdb \
// RUN:   -x hip --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK %s

// CHECK: {{.*}}clang{{.*}}"-triple" "amdgcn-amd-amdhsa"
// CHECK-NOT: "-disable-O0-optnone"
// CHECK-NOT: "-disable-O0-noinline"
// CHECK-DAG: "-debug-info-kind=line-tables-only"
// CHECK-DAG: "-debugger-tuning=gdb"
// CHECK-NOT: "-disable-O0-optnone"
// CHECK-NOT: "-disable-O0-noinline"
// CHECK: {{.*}}llc{{.*}}"-mtriple=amdgcn-amd-amdhsa"
// CHECK-NOT: "-disable-O0-optnone"
// CHECK-NOT: "-disable-O0-noinline"
// CHECK: "-amdgpu-spill-cfi-saved-regs"
// CHECK-NOT: "-disable-O0-optnone"
// CHECK-NOT: "-disable-O0-noinline"
// CHECK-LABEL: clang-offload-bundler
// CHECK: {{.*}}clang{{.*}}"-triple" "x86_64-unknown-linux-gnu"
// CHECK-NOT: "-disable-O0-optnone"
// CHECK-NOT: "-disable-O0-noinline"
// CHECK-NOT: "-amdgpu-spill-cfi-saved-regs"
// CHECK-DAG: "-debug-info-kind=limited"
// CHECK-DAG: "-debugger-tuning=gdb"
// CHECK-NOT: "-disable-O0-optnone"
// CHECK-NOT: "-disable-O0-noinline"
// CHECK-NOT: "-amdgpu-spill-cfi-saved-regs"
