// Check that -ggdb implies the right options and is composable

// Check for the expected effects of -ggdb
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -ggdb %s 2>&1 | FileCheck -check-prefix=CHECK-SIMPLE %s
// CHECK-SIMPLE: "-cc1"
// CHECK-SIMPLE-NOT "-disable-O0-optnone"
// CHECK-SIMPLE-NOT "-disable-O0-noinline"
// CHECK-SIMPLE-NOT: "-debug-info-kind=line-tables-only"
// CHECK-SIMPLE-DAG: "-mllvm" "-amdgpu-spill-cfi-saved-regs"
// CHECK-SIMPLE-DAG: "-mllvm" "-disable-dwarf-locations"
// CHECK-SIMPLE-DAG: "-debugger-tuning=gdb"
// CHECK-SIMPLE-NOT "-disable-O0-optnone"
// CHECK-SIMPLE-NOT "-disable-O0-noinline"
// CHECK-SIMPLE-NOT: "-debug-info-kind=line-tables-only"

// Check that a debug-related option which does not affect the debug-info-kind
// is still composable with -ggdb
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -gdwarf-2 -ggdb %s 2>&1 | FileCheck -check-prefix=CHECK-DWARF2 %s
// CHECK-DWARF2: "-cc1"
// CHECK-DWARF2-NOT: "-disable-O0-optnone"
// CHECK-DWARF2-NOT: "-disable-O0-noinline"
// CHECK-DWARF2-NOT: "-debug-info-kind=line-tables-only"
// CHECK-DWARF2-DAG: "-mllvm" "-amdgpu-spill-cfi-saved-regs"
// CHECK-DWARF2-DAG: "-mllvm" "-disable-dwarf-locations"
// CHECK-DWARF2-DAG: "-debugger-tuning=gdb"
// CHECK-DWARF2-DAG: "-dwarf-version=2"
// CHECK-DWARF2-NOT: "-disable-O0-optnone"
// CHECK-DWARF2-NOT: "-disable-O0-noinline"
// CHECK-DWARF2-NOT: "-debug-info-kind=line-tables-only"

// Check that -ggdb does not affect the debug-info-kind for AMDGPU.
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -g -ggdb %s 2>&1 | FileCheck -check-prefix=CHECK-GBEFORE %s
// CHECK-GBEFORE: "-cc1"
// CHECK-GBEFORE: "-debug-info-kind=limited"
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -ggdb -g %s 2>&1 | FileCheck -check-prefix=CHECK-GAFTER %s
// CHECK-GAFTER: "-cc1"
// CHECK-GAFTER: "-debug-info-kind=limited"

// Check that -ggdb composes with other tuning options
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -glldb -ggdb %s 2>&1 | FileCheck -check-prefix=CHECK-LLDBBEFORE %s
// CHECK-LLDBBEFORE: "-cc1"
// CHECK-LLDBBEFORE-NOT: "-disable-O0-optnone"
// CHECK-LLDBBEFORE-NOT: "-disable-O0-noinline"
// CHECK-LLDBBEFORE-NOT: "-debug-info-kind=line-tables-only"
// CHECK-LLDBBEFORE-DAG: "-mllvm" "-amdgpu-spill-cfi-saved-regs"
// CHECK-LLDBBEFORE-DAG: "-mllvm" "-disable-dwarf-locations"
// CHECK-LLDBBEFORE-DAG: "-debugger-tuning=gdb"
// CHECK-LLDBBEFORE-NOT: "-disable-O0-optnone"
// CHECK-LLDBBEFORE-NOT: "-disable-O0-noinline"
// CHECK-LLDBBEFORE-NOT: "-debug-info-kind=line-tables-only"
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -ggdb -glldb %s 2>&1 | FileCheck -check-prefix=CHECK-LLDBAFTER %s
// CHECK-LLDBAFTER: "-cc1"
// CHECK-LLDBAFTER-DAG: "-debugger-tuning=lldb"
