// Verify clang emits the "debugger-tuning" module flag whenever a tuning is
// specified.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debugger-tuning=gdb -emit-llvm %s -o - | FileCheck -check-prefix=GDB %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debugger-tuning=lldb -emit-llvm %s -o - | FileCheck -check-prefix=LLDB %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debugger-tuning=sce -emit-llvm %s -o - | FileCheck -check-prefix=SCE %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -debugger-tuning=dbx -emit-llvm %s -o - | FileCheck -check-prefix=DBX %s

// A requested tuning that matches the target default is still recorded
// RUN: %clang_cc1 -triple x86_64-apple-darwin -debugger-tuning=lldb -emit-llvm %s -o - | FileCheck -check-prefix=LLDB %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck -check-prefix=DEFAULT %s

void f(void) {}

// GDB: !{i32 2, !"debugger-tuning", !"gdb"}
// LLDB: !{i32 2, !"debugger-tuning", !"lldb"}
// SCE: !{i32 2, !"debugger-tuning", !"sce"}
// DBX: !{i32 2, !"debugger-tuning", !"dbx"}
// DEFAULT-NOT: "debugger-tuning"
