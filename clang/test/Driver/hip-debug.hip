// REQUIRES: amdgpu-registered-target, x86-registered-target

// Check that -g correctly differentiates device/host code, and that the
// -amdgpu-spill-cfi-saved-regs and -gheterogeneous-dwarf options are
// supplied during device code-gen (i.e. on the device -cc1 command line) but
// not for the host.

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpulib -nogpuinc -g \
// RUN:   -x hip --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=DEFAULT %s

// DEFAULT: {{.*}}clang{{.*}}"-triple" "amdgcn-amd-amdhsa"
// DEFAULT-NOT: "-disable-O0-optnone"
// DEFAULT-NOT: "-debug-info-kind=line-tables-only"
// DEFAULT-DAG: "-debug-info-kind=constructor"
// DEFAULT-DAG: "-debugger-tuning=gdb"
// DEFAULT-DAG: "-mllvm" "-amdgpu-spill-cfi-saved-regs"
// DEFAULT-DAG: "-gheterogeneous-dwarf=diexpression"
// DEFAULT-NOT: "-disable-O0-optnone"
// DEFAULT-NOT: "-debug-info-kind=line-tables-only"
// DEFAULT-LABEL: clang-offload-bundler
// DEFAULT: {{.*}}clang{{.*}}"-triple" "x86_64-unknown-linux-gnu"
// DEFAULT-NOT: "-disable-O0-optnone"
// DEFAULT-NOT: "-debug-info-kind=line-tables-only"
// DEFAULT-NOT: "-amdgpu-spill-cfi-saved-regs"
// DEFAULT-NOT: "-gheterogeneous-dwarf"
// DEFAULT-DAG: "-debug-info-kind=constructor"
// DEFAULT-DAG: "-debugger-tuning=gdb"
// DEFAULT-NOT: "-disable-O0-optnone"
// DEFAULT-NOT: "-debug-info-kind=line-tables-only"
// DEFAULT-NOT: "-amdgpu-spill-cfi-saved-regs"
// DEFAULT-NOT: "-gheterogeneous-dwarf"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpulib -nogpuinc -g \
// RUN:   -x hip --cuda-gpu-arch=gfx900 -fgpu-rdc %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=RDC %s

// RDC: {{.*}}clang{{.*}}"-triple" "amdgcn-amd-amdhsa"
// RDC-DAG: "-mllvm" "-amdgpu-spill-cfi-saved-regs"
// RDC-DAG: "-gheterogeneous-dwarf=diexpression"
