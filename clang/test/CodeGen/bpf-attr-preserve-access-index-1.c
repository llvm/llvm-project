// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

#define __reloc__ __attribute__((preserve_access_index))

// test simple member access and initial struct with non-zero stride access
struct s1 {
  int a;
  union {
   int b;
   int c;
  };
} __reloc__;
typedef struct s1 __s1;

int test(__s1 *arg) {
  return arg->a + arg[1].b;
}

// CHECK: call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 0)
// CHECK: call %struct.s1* @llvm.preserve.array.access.index.p0s_struct.s1s.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 1)
// CHECK: call %union.anon* @llvm.preserve.struct.access.index.p0s_union.anons.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK: call %union.anon* @llvm.preserve.union.access.index.p0s_union.anons.p0s_union.anons(%union.anon* %{{[0-9a-z]+}}, i32 0)
