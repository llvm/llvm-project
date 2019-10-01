// RUN: %clangxx -target x86_64-pc-linux -gsplit-dwarf -fsplit-dwarf-inlining \
// RUN:   -c %s -o %t
// RUN: %lldb %t -o "breakpoint set -n foo" -b | FileCheck %s

// CHECK: Breakpoint 1: 2 locations

__attribute__((always_inline)) int foo(int x) { return x; }
int bar(int x) { return foo(x); }
