// RUN: %clang_cc1 %s -triple x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s
// The store of p.y into the temporary was not
// getting extended to 32 bits, so uninitialized
// bits of the temporary were used.  7366161.
struct foo {
  char x:8;
  signed int y:24;
};
int bar(struct foo p, int x) {
// CHECK: bar
// CHECK: %[[val:.*]] = load i32, ptr {{.*}}
// CHECK-NEXT:          ashr i32 %[[val]]
// CHECK:             = load i32, ptr {{.*}}
// CHECK:             = load i32, ptr {{.*}}
// CHECK: %[[val:.*]] = load i32, ptr {{.*}}
// CHECK-NEXT:          ashr i32 %[[val]]
  x = (p.y > x ? x : p.y);
  return x;
// CHECK: ret
}
