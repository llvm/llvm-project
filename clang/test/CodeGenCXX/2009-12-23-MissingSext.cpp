// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// The store of p.y into the temporary was not
// getting extended to 32 bits, so uninitialized
// bits of the temporary were used.  7366161.
struct foo {
  char x:8;
  signed int y:24;
};
int bar(struct foo p, int x) {
// CHECK: bar
// CHECK: and {{.*}} 16777215
// CHECK: and {{.*}} 16777215
  x = (p.y > x ? x : p.y);
  return x;
// CHECK: ret
}
