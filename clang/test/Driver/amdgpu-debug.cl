// Check that -ggdb implies the right options and is composable

// Check for the expected effects of -g and -ggdb for AMDGCN
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -g %s 2>&1 | FileCheck -check-prefix=CHECK-SIMPLE %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -ggdb %s 2>&1 | FileCheck -check-prefix=CHECK-SIMPLE %s
// CHECK-SIMPLE: "-cc1"
// CHECK-SIMPLE-NOT: "-disable-O0-optnone"
// CHECK-SIMPLE-NOT: "-debug-info-kind=line-tables-only"
// CHECK-SIMPLE-DAG: "-mllvm" "-amdgpu-spill-cfi-saved-regs"
// CHECK-SIMPLE-DAG: "-gheterogeneous-dwarf=diexpression"
// CHECK-SIMPLE-DAG: "-debugger-tuning=gdb"
// CHECK-SIMPLE-NOT: "-disable-O0-optnone"
// CHECK-SIMPLE-NOT: "-debug-info-kind=line-tables-only"

// Check that -gheterogeneous-dwarf is not enabled for AMDGCN when debug information is not enabled
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm %s 2>&1 | FileCheck -check-prefix=CHECK-NO-G %s
// CHECK-NO-G: "-cc1"
// CHECK-NO-G-NOT: "-amdgpu-spill-cfi-saved-regs"
// CHECK-NO-G-NOT: "-gheterogeneous-dwarf"

// Check that -gheterogeneous-dwarf can be enabled for non-AMDGCN
// RUN: %clang -### -target x86_64-linux-gnu -x cl -c -nogpuinc -nogpulib  -emit-llvm -gheterogeneous-dwarf %s 2>&1 | FileCheck -check-prefix=CHECK-EXPLICIT-HETEROGENEOUS %s
// CHECK-EXPLICIT-HETEROGENEOUS: "-cc1"
// CHECK-EXPLICIT-HETEROGENEOUS: "-gheterogeneous-dwarf=diexpression"

// Check that -gheterogeneous-dwarf can be disabled for AMDGCN
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -g -gno-heterogeneous-dwarf %s 2>&1 | FileCheck -check-prefix=CHECK-NO-HETEROGENEOUS %s
// CHECK-NO-HETEROGENEOUS: "-cc1"
// CHECK-NO-HETEROGENEOUS: "-gheterogeneous-dwarf=disabled"

// Check that -gheterogeneous-dwarf= works for disabling
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -g -gheterogeneous-dwarf=disabled %s 2>&1 | FileCheck -check-prefix=CHECK-DISABLED %s
// CHECK-DISABLED: "-cc1"
// CHECK-DISABLED: "-gheterogeneous-dwarf=disabled"

// Check that -gheterogeneous-dwarf= works for diexpr
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -g -gheterogeneous-dwarf=diexpr %s 2>&1 | FileCheck -check-prefix=CHECK-DIEXPR %s
// CHECK-DIEXPR: "-cc1"
// CHECK-DIEXPR: "-gheterogeneous-dwarf=diexpr"

// Check that -gheterogeneous-dwarf= works for diexpression
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -g -gheterogeneous-dwarf=diexpression %s 2>&1 | FileCheck -check-prefix=CHECK-DIEXPRESSION %s
// CHECK-DIEXPRESSION: "-cc1"
// CHECK-DIEXPRESSION: "-gheterogeneous-dwarf=diexpression"

// Check that -gheterogeneous-dwarf= fails for unknown option
// RUN: not %clang -target amdgcn-amd-amdhsa -x cl -c -nogpuinc -nogpulib  -emit-llvm -g -gheterogeneous-dwarf=unknown %s 2>&1 | FileCheck -check-prefix=CHECK-UNKNOWN %s
// CHECK-UNKNOWN: error: invalid value
