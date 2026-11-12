// RUN: %clang %s -target bpfeb -x c -emit-llvm -S -g -O2 -o - | FileCheck %s
// RUN: %clang %s -target bpfel -x c -emit-llvm -S -g -O2 -o - | FileCheck %s

struct t {
  int i:1;
  int j:2;
  union {
   int a;
   int b;
  } c[4];
};

#define _(x) (x)

const void *test(struct t *arg) {
  return _(&arg->c[3].b);
}

// CHECK-NOT: llvm.preserve.struct.access.index
// CHECK-NOT: llvm.preserve.array.access.index
// CHECK-NOT: llvm.preserve.union.access.index
// CHECK-NOT: __builtin_preserve_access_index
