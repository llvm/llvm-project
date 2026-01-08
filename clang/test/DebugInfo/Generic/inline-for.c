// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// Check that clang emits Debug location in the phi instruction

int func(int n) {
  int a;
  for(a = 10; a>0 && n++; a--);
  return n;
}

// CHECK: land.end:
// CHECK-NEXT: {{.*}} = phi i1 {{.*}} !dbg ![[DbgLoc:[0-9]+]]

// CHECK: ![[DbgLoc]] = !DILocation(line: 0
