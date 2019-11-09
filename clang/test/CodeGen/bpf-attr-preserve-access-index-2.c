// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -emit-llvm -S -g %s -o - | FileCheck %s

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

// CHECK: call [3 x i32]* @llvm.preserve.struct.access.index.p0a3i32.p0s_struct.s1s(%struct.s1* %{{[0-9a-z]+}}, i32 0, i32 0)
// CHECK: call i32* @llvm.preserve.array.access.index.p0i32.p0a3i32([3 x i32]* %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK: call %union.anon* @llvm.preserve.struct.access.index.p0s_union.anons.p0s_struct.s1s(%struct.s1* %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK: call %union.anon* @llvm.preserve.union.access.index.p0s_union.anons.p0s_union.anons(%union.anon* %{{[0-9a-z]+}}, i32 1)
// CHECK: call i32* @llvm.preserve.array.access.index.p0i32.p0a4i32([4 x i32]* %{{[0-9a-z]+}}, i32 1, i32 2)
