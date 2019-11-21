// Check that -grocgdb implies the right options and is composable

// Check for the expected effects of -grocgdb
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -grocgdb %s 2>&1 | FileCheck -check-prefix=CHECK-SIMPLE %s
// CHECK-SIMPLE: "-cc1"
// CHECK-SIMPLE-DAG: "-debug-info-kind=line-tables-only"
// CHECK-SIMPLE-DAG: "-disable-O0-optnone"
// CHECK-SIMPLE-DAG: "-disable-O0-noinline"
// CHECK-SIMPLE-DAG: "-debugger-tuning=gdb"

// Check that a debug-related option which does not affect the debug-info-kind
// is still composable with -grocgdb
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -gdwarf-2 -grocgdb %s 2>&1 | FileCheck -check-prefix=CHECK-DWARF2 %s
// CHECK-DWARF2: "-cc1"
// CHECK-DWARF2-DAG: "-debug-info-kind=line-tables-only"
// CHECK-DWARF2-DAG: "-disable-O0-optnone"
// CHECK-DWARF2-DAG: "-disable-O0-noinline"
// CHECK-DWARF2-DAG: "-debugger-tuning=gdb"
// CHECK-DWARF2-DAG: "-dwarf-version=2"

// Check that options which affect the debug-info-kind are silently ignored
// when -grocgdb is in effect, even when they appear after it. This behavior
// may change in the future
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -g -grocgdb %s 2>&1 | FileCheck -check-prefix=CHECK-GBEFORE %s
// CHECK-GBEFORE: "-cc1"
// CHECK-GBEFORE: "-debug-info-kind=line-tables-only"
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -grocgdb -g %s 2>&1 | FileCheck -check-prefix=CHECK-GAFTER %s
// CHECK-GAFTER: "-cc1"
// CHECK-GAFTER: "-debug-info-kind=line-tables-only"

// Check that -grocgdb composes with other tuning options
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -glldb -grocgdb %s 2>&1 | FileCheck -check-prefix=CHECK-LLDBBEFORE %s
// CHECK-LLDBBEFORE: "-cc1"
// CHECK-LLDBBEFORE-DAG: "-debug-info-kind=line-tables-only"
// CHECK-LLDBBEFORE-DAG: "-disable-O0-optnone"
// CHECK-LLDBBEFORE-DAG: "-disable-O0-noinline"
// CHECK-LLDBBEFORE-DAG: "-debugger-tuning=gdb"
// RUN: %clang -### -target amdgcn-amd-amdhsa -x cl -c -emit-llvm -grocgdb -glldb %s 2>&1 | FileCheck -check-prefix=CHECK-LLDBAFTER %s
// CHECK-LLDBAFTER: "-cc1"
// CHECK-LLDBAFTER-DAG: "-debugger-tuning=lldb"
