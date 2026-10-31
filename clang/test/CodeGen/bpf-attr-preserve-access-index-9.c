// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

#define __reloc__ __attribute__((preserve_access_index))

// Every subscript of a multi-dimensional array member is preserved, so the
// record behind the array stays in the same access chain.
struct s1 {
  long a;
  int b;
} __reloc__;
typedef const struct s1 __s1;
typedef struct s1 arr2_t[2][3];
typedef struct s1 row_t[3];

union u1 {
  int c;
  __s1 d;
} __reloc__;

struct s2 {
  int e;
  struct s1 f[2][3];
  volatile union u1 g[2][3];
  struct s1 (*h)[3];
  struct s1 *i[3];
  const arr2_t j;
  row_t k[2];
  struct s1 l[2][3][4];
} __reloc__;
typedef struct s2 __s2;

union u2 {
  int l;
  __s1 m[2][3];
} __reloc__;
typedef union u2 u2row_t[2];

struct s3 {
  int n;
  const u2row_t o[3];
} __reloc__;

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

int test2(__s2 *arg) { return arg->g[1][2].d.b; }

// CHECK-LABEL: @test2
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 2, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %union.u1]]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %union.u1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

int test3(union u2 *arg) { return arg->m[1][2].b; }

// CHECK-LABEL: @test3
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.s1]]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

int test4(__s2 *arg) { return arg->h[1][2].b; }

// CHECK-LABEL: @test4
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 3, i32 3)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 0, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

int test5(__s2 *arg) { return arg->i[1][2].b; }

// CHECK-LABEL: @test5
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 4, i32 4)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x ptr]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

int test6(__s2 *arg) { return arg->j[1][2].b; }

// CHECK-LABEL: @test6
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 5, i32 5)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.s1]]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

int test7(__s2 *arg) { return arg->k[1][2].b; }

// CHECK-LABEL: @test7
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 6, i32 6)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.s1]]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

// struct -> array -> union -> array -> struct
int test8(struct s3 *arg) { return arg->o[2][1].m[1][2].b; }

// CHECK-LABEL: @test8
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s3) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x [2 x %union.u2]]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x %union.u2]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.s1]]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

// A variable index cannot be preserved, but later constant subscripts can.
int test9(__s2 *arg, int n) { return arg->f[n][2].b; }

// CHECK-LABEL: @test9
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK:       getelementptr inbounds [2 x [3 x %struct.s1]], ptr %{{[0-9a-z]+}}, i64 0, i64 %{{[0-9a-z]+}}
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

// Parentheses do not change the access chain.
int test10(__s2 *arg) { return (arg->f[1])[2].b; }

// CHECK-LABEL: @test10
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.s1]]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

int test11(__s2 *arg) { return (arg->f)[1][2].b; }

// CHECK-LABEL: @test11
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.s1]]) %{{[0-9a-z]+}}, i32 1, i32 1)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)

// A variable index in a 3-D array member: the constant subscripts after it are
// still preserved.
int test12(__s2 *arg, int n) { return arg->l[n][2][3].b; }

// CHECK-LABEL: @test12
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 7, i32 7)
// CHECK:       getelementptr inbounds [2 x [3 x [4 x %struct.s1]]], ptr %{{[0-9a-z]+}}, i64 0, i64 %{{[0-9a-z]+}}
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x [4 x %struct.s1]]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([4 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 3)
// CHECK-NOT:   getelementptr
// CHECK:       call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1)
