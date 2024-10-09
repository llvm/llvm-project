// RUN: %clang_cc1 -emit-llvm -O2 -debug-info-kind=limited -fextend-lifetimes -o - %s | FileCheck %s

// Clang adjusts the line numbers of returns based on the line numbers of
// dominating stores to %retval; we test that fake use intrinsics do not affect
// this, and the return is given the correct line.

// CHECK: define{{.*}}@main
// CHECK: ret i32{{.*}}!dbg ![[MDINDEX:[0-9]*]]
// CHECK: ![[MDINDEX]] = !DILocation(line: [[# @LINE + 5]]
int main()
{
  volatile int a = 1;
  int b = a + 2;
  return b;
}
