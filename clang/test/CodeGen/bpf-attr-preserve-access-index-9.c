// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

#define __reloc__ __attribute__((preserve_access_index))

struct s1 {
  long a;
  int b;
} __reloc__;

struct s2 {
  int e;
  struct s1 f[2][3];
} __reloc__;
typedef struct s2 __s2;

int test1(__s2 *arg) { return arg->f[1][2].b; }

// CHECK-LABEL: @test1
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.s1]]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)
