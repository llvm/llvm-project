// RUN: %clang_cc1 -O1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s
// RUN: %clang_cc1 -O1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=line-directives-only %s -o - | FileCheck %s

// Inserting lifetime markers should not affect debuginfo

extern int x;

// CHECK-LABEL: define i32 @f
int f() {
  int *p = &x;
// CHECK: ret i32 %{{.*}}, !dbg [[DI:![0-9]*]]
// CHECK: [[DI]] = !DILocation(line: [[@LINE+1]]
  return *p;
}
