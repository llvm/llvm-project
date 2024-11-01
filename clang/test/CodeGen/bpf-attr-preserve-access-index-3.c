// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

#define __reloc__ __attribute__((preserve_access_index))

// chain of records, all with attributes
struct s1 {
  int c;
} __reloc__;
typedef struct s1 __s1;

struct s2 {
  union {
    __s1 b[3];
  };
} __reloc__;
typedef struct s2 __s2;

struct s3 {
  __s2 a;
} __reloc__;
typedef struct s3 __s3;

int test(__s3 *arg) {
  return arg->a.b[2].c;
}

// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s3) %{{[0-9a-z]+}}, i32 0, i32 0)
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 0, i32 0)
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 0)
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 0)
