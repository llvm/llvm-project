// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

#define __reloc__ __attribute__((preserve_access_index))

// test array access
struct s1 {
  int a[3];
  union {
   int b;
   int c[4];
  };
} __reloc__;
typedef struct s1 __s1;

int test(__s1 *arg) {
  return arg->a[2] + arg->c[2];
}

// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 0)
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x i32]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 1)
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([4 x i32]) %{{[0-9a-z]+}}, i32 1, i32 2)
