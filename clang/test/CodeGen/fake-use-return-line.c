// RUN: %clang_cc1 -emit-llvm -O2 -debug-info-kind=limited -fextend-lifetimes -o - %s | FileCheck %s
int main()
{
  volatile int a = 1;
  int b = a + 2;
  return b;
}
// CHECK: define{{.*}}@main
// CHECK: ret i32{{.*}}!dbg ![[MDINDEX:[0-9]*]]
// CHECK: ![[MDINDEX]] = !DILocation(line: 6
