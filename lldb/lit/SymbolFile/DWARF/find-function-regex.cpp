// REQUIRES: lld

// RUN: %clang %s -g -c -o %t.o --target=x86_64-pc-linux -mllvm -accel-tables=Disable
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=f.o --regex --find=function %t | FileCheck %s
//
// RUN: %clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=f.o --regex --find=function %t | FileCheck %s

// RUN: %clang %s -g -c -o %t.o --target=x86_64-pc-linux -mllvm -accel-tables=Dwarf
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=f.o --regex --find=function %t | FileCheck %s

// CHECK: Found 3 functions:
// CHECK-DAG: name = "foo()", mangled = "_Z3foov"
// CHECK-DAG: name = "ffo()", mangled = "_Z3ffov"
// CHECK-DAG: name = "bar::foo()", mangled = "_ZN3bar3fooEv"

void foo() {}
void ffo() {}
namespace bar {
void foo() {}
} // namespace bar
void fof() {}

extern "C" void _start() {}
