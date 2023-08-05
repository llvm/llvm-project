// Check that -ggdb implies the right options and is composable

// Check for the expected effects of -g and -ggdb for AMDGCN
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -g %s 2>&1 | FileCheck -check-prefix=CHECK-SIMPLE %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -ggdb %s 2>&1 | FileCheck -check-prefix=CHECK-SIMPLE %s
// CHECK-SIMPLE: "-cc1"
// CHECK-SIMPLE-NOT: "-disable-O0-optnone"
// CHECK-SIMPLE-NOT: "-debug-info-kind=line-tables-only"
// CHECK-SIMPLE-DAG: "-mllvm" "-amdgpu-spill-cfi-saved-regs"
// CHECK-SIMPLE-DAG: "-gheterogeneous-dwarf"
// CHECK-SIMPLE-DAG: "-debugger-tuning=gdb"
// CHECK-SIMPLE-NOT: "-disable-O0-optnone"
// CHECK-SIMPLE-NOT: "-debug-info-kind=line-tables-only"

// Check that -gheterogeneous-dwarf is not enabled for AMDGCN when debug information is not enabled
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm %s 2>&1 | FileCheck -check-prefix=CHECK-NO-G %s
// CHECK-NO-G: "-cc1"
// CHECK-NO-G-NOT: "-amdgpu-spill-cfi-saved-regs"
// CHECK-NO-G-NOT: "-gheterogeneous-dwarf"

// Check that -gheterogeneous-dwarf can be disabled for AMDGCN
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -g -gno-heterogeneous-dwarf %s 2>&1 | FileCheck -check-prefix=CHECK-NO-HETEROGENEOUS %s
// CHECK-NO-HETEROGENEOUS: "-cc1"
// CHECK-NO-HETEROGENEOUS-NOT: "-gheterogeneous-dwarf"
