// REQUIRES: bpf-registered-target
// RUN: %clang -Xclang -cl-std=CL2.0 -emit-llvm -g -gheterogeneous-dwarf=diexpr -O0 -S -nogpulib -target bpf-linux-gnu -o - %s | FileCheck %s

// FIXME: Currently just testing that we don't crash; test for the absense
// of meaningful debug information for the extern is to identify this test
// to update/replace when this is implemented.

// CHECK-NOT: DIGlobalVariable

extern char ch;
int test() {
  return ch;
}
