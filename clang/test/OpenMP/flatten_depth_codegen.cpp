// Check code generation for the 'depth' clause on '#pragma omp flatten'.
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=61 -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

extern "C" void body(int, int, int);

// CHECK-LABEL: define {{.*}}void @foo3(
// CHECK:   %.flatten.iv = alloca i64
// CHECK:   %.flatten.iv.0 = alloca i32
// CHECK:   %.flatten.iv.1 = alloca i32
// CHECK:   %.flatten.iv.2 = alloca i32
// CHECK:   %[[CIV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[T01:.+]] = mul nsw i64 %{{.+}}, %{{.+}}
// CHECK:   %[[TALL:.+]] = mul nsw i64 %[[T01]], %{{.+}}
// CHECK:   icmp slt i64 %[[CIV]], %[[TALL]]
// CHECK:   %[[D0IV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[D0M:.+]] = mul nsw i64 %{{.+}}, %{{.+}}
// CHECK:   %[[D0:.+]] = sdiv i64 %[[D0IV]], %[[D0M]]
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.0
// CHECK:   %[[D1IV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[D1:.+]] = sdiv i64 %[[D1IV]], %{{.+}}
// CHECK:   %[[R1:.+]] = srem i64 %[[D1]], %{{.+}}
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.1
// CHECK:   %[[R2IV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[R2:.+]] = srem i64 %[[R2IV]], %{{.+}}
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.2
// CHECK:   call void @body(
extern "C" void foo3(int n, int m, int p) {
#pragma omp flatten depth(3)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      for (int k = 0; k < p; ++k)
        body(i, j, k);
}

// CHECK-LABEL: define {{.*}}void @foo2(
// CHECK:   %.flatten.iv = alloca i64
// CHECK:   %.flatten.iv.0 = alloca i32
// CHECK:   %.flatten.iv.1 = alloca i32
// CHECK:   %[[CIV2:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[M:.+]] = mul nsw i64 %{{.+}}, %{{.+}}
// CHECK:   icmp slt i64 %[[CIV2]], %[[M]]
// CHECK:   sdiv i64 %{{.+}}, %{{.+}}
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.0
// CHECK:   srem i64 %{{.+}}, %{{.+}}
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.1
extern "C" void foo2(int n, int m) {
#pragma omp flatten depth(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      body(i, j, 0);
}

// CHECK-LABEL: define {{.*}}void @foo1(
// CHECK:   %.flatten.iv = alloca i32
// CHECK:   %.flatten.iv.0 = alloca i32
// CHECK: for.body:
// CHECK:   %[[FV:.+]] = load i32, ptr %.flatten.iv,
// CHECK:   store i32 %[[FV]], ptr %.flatten.iv.0,
// CHECK-NOT: sdiv
// CHECK-NOT: srem
// CHECK:   call void @body(
extern "C" void foo1(int n) {
#pragma omp flatten depth(1)
  for (int i = 0; i < n; ++i)
    body(i, 0, 0);
}

// CHECK-LABEL: define {{.*}}void @foo2partial(
// CHECK:   %.flatten.iv = alloca i64
// CHECK:   %.flatten.iv.0 = alloca i32
// CHECK:   %.flatten.iv.1 = alloca i32
// CHECK-NOT: %.flatten.iv.2 = alloca
// CHECK:   %[[CIV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[M:.+]] = mul nsw i64 %{{.+}}, %{{.+}}
// CHECK:   icmp slt i64 %[[CIV]], %[[M]]
// CHECK:   sdiv i64 %{{.+}}, %{{.+}}
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.0
// CHECK:   srem i64 %{{.+}}, %{{.+}}
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.1
// CHECK:   br label %for.cond{{.*}}
// CHECK: for.cond{{.*}}:
// CHECK:   icmp slt i32 %{{.+}}, 5
extern "C" void foo2partial(int n, int m) {
#pragma omp flatten depth(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      for (int k = 0; k < 5; ++k)
        body(i, j, k);
}

// A value-dependent 'depth' (the argument is a non-type template parameter) is
// deferred until instantiation, where the concrete depth drives the flattening.

// CHECK-LABEL: define {{.*}}void @_Z11tmpl_depth3ILi3EEviii(
// CHECK:   %.flatten.iv = alloca i64
// CHECK:   %.flatten.iv.0 = alloca i32
// CHECK:   %.flatten.iv.1 = alloca i32
// CHECK:   %.flatten.iv.2 = alloca i32
// CHECK:   %[[CIV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[T01:.+]] = mul nsw i64 %{{.+}}, %{{.+}}
// CHECK:   %[[TALL:.+]] = mul nsw i64 %[[T01]], %{{.+}}
// CHECK:   icmp slt i64 %[[CIV]], %[[TALL]]
// CHECK:   %[[D0IV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[D0M:.+]] = mul nsw i64 %{{.+}}, %{{.+}}
// CHECK:   %[[D0:.+]] = sdiv i64 %[[D0IV]], %[[D0M]]
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.0
// CHECK:   %[[D1IV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[D1:.+]] = sdiv i64 %[[D1IV]], %{{.+}}
// CHECK:   %[[R1:.+]] = srem i64 %[[D1]], %{{.+}}
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.1
// CHECK:   %[[R2IV:.+]] = load i64, ptr %.flatten.iv,
// CHECK:   %[[R2:.+]] = srem i64 %[[R2IV]], %{{.+}}
// CHECK:   store i32 %{{.+}}, ptr %.flatten.iv.2
// CHECK:   call void @body(
template <int D>
void tmpl_depth3(int n, int m, int p) {
#pragma omp flatten depth(D)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      for (int k = 0; k < p; ++k)
        body(i, j, k);
}

// A value-dependent 'depth' that instantiates to depth(1) on a single loop is
// well-formed: the shallower-than-default nest is accepted because validation is
// deferred to instantiation. The single loop needs no div/mod recovery.

// CHECK-LABEL: define {{.*}}void @_Z11tmpl_depth1ILi1EEvi(
// CHECK:   %.flatten.iv = alloca i32
// CHECK:   %.flatten.iv.0 = alloca i32
// CHECK: for.body:
// CHECK:   %[[FV:.+]] = load i32, ptr %.flatten.iv,
// CHECK:   store i32 %[[FV]], ptr %.flatten.iv.0,
// CHECK-NOT: sdiv
// CHECK-NOT: srem
// CHECK:   call void @body(
template <int D>
void tmpl_depth1(int n) {
#pragma omp flatten depth(D)
  for (int i = 0; i < n; ++i)
    body(i, 0, 0);
}

void inst() {
  tmpl_depth3<3>(1, 1, 1);
  tmpl_depth1<1>(1);
}

#endif
