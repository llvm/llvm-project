// Pipe stderr to FileCheck since we're checking for a warning
// RUN: %clang -gcodeview -g -emit-llvm -S %s -o - 2>&1 | FileCheck %s


__attribute__((debug_transparent)) 
void foo(void) {}

// CHECK: warning: 'debug_transparent' attribute is ignored since it is only supported by DWARF

