// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Check that -g correctly differentiates device/host code, and that the
// -amdgpu-spill-cfi-saved-regs and -gheterogeneous-dwarf options are
// supplied during actual code-gen (i.e. in the llc command-line for the
// device in the normal case, and the lld command-line in the RDC case).

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpulib -g \
// RUN:   -x hip --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=DEFAULT %s

// DEFAULT: {{.*}}clang{{.*}}"-triple" "amdgcn-amd-amdhsa"
// DEFAULT-NOT: "-disable-O0-optnone"
// DEFAULT-NOT: "-debug-info-kind=line-tables-only"
// DEFAULT-NOT: "-disable-dwarf-locations"
// DEFAULT-DAG: "-debug-info-kind=constructor"
// DEFAULT-DAG: "-debugger-tuning=gdb"
// DEFAULT-DAG: "-mllvm" "-amdgpu-spill-cfi-saved-regs"
// DEFAULT-DAG: "-gheterogeneous-dwarf"
// DEFAULT-NOT: "-disable-O0-optnone"
// DEFAULT-NOT: "-debug-info-kind=line-tables-only"
// DEFAULT-NOT: "-disable-dwarf-locations"
// DEFAULT-LABEL: clang-offload-bundler
// DEFAULT: {{.*}}clang{{.*}}"-triple" "x86_64-unknown-linux-gnu"
// DEFAULT-NOT: "-disable-O0-optnone"
// DEFAULT-NOT: "-debug-info-kind=line-tables-only"
// DEFAULT-NOT: "-amdgpu-spill-cfi-saved-regs"
// DEFAULT-NOT: "-disable-dwarf-locations"
// DEFAULT-NOT: "-gheterogeneous-dwarf"
// DEFAULT-DAG: "-debug-info-kind=constructor"
// DEFAULT-DAG: "-debugger-tuning=gdb"
// DEFAULT-NOT: "-disable-O0-optnone"
// DEFAULT-NOT: "-debug-info-kind=line-tables-only"
// DEFAULT-NOT: "-amdgpu-spill-cfi-saved-regs"
// DEFAULT-NOT: "-disable-dwarf-locations"
// DEFAULT-NOT: "-gheterogeneous-dwarf"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -nogpulib -g \
// RUN:   -x hip --cuda-gpu-arch=gfx900 -fgpu-rdc %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=RDC %s

// RDC: {{.*}}lld{{.*}} "-plugin-opt=mcpu=gfx900"
// RDC-SAME: "-plugin-opt=-amdgpu-spill-cfi-saved-regs"
// RDC-NOT: "-plugin-opt=-disable-dwarf-locations"
// RDC-NOT: "-plugin-opt=-gheterogeneous-dwarf"
