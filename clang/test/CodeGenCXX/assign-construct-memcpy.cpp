// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -emit-llvm -o - -std=c++11 %s -DPOD | FileCheck %s -check-prefix=CHECK-POD
// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -emit-llvm -o - -std=c++11 %s | FileCheck %s -check-prefix=CHECK-NONPOD

// Declare the reserved placement operators.
typedef __typeof__(sizeof(0)) size_t;
void *operator new(size_t, void*) throw();
void operator delete(void*, void*) throw();
void *operator new[](size_t, void*) throw();
void operator delete[](void*, void*) throw();
template<typename T> T &&move(T&);

struct foo {
#ifndef POD
  foo() {} // non-POD
#endif
  void *a, *b;
  bool c;
};

// It is not legal to copy the tail padding in all cases, but if it is it can
// yield better codegen.

foo *test1(void *f, const foo &x) {
  return new (f) foo(x);
// CHECK-POD: test1
// CHECK-POD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1

// CHECK-NONPOD: test1
// CHECK-NONPOD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1
}

foo *test2(const foo &x) {
  return new foo(x);
// CHECK-POD: test2
// CHECK-POD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1

// CHECK-NONPOD: test2
// CHECK-NONPOD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1
}

foo test3(const foo &x) {
  foo f = x;
  return f;
// CHECK-POD: test3
// CHECK-POD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1

// CHECK-NONPOD: test3
// CHECK-NONPOD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1
}

foo *test4(foo &&x) {
  return new foo(x);
// CHECK-POD: test4
// CHECK-POD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1

// CHECK-NONPOD: test4
// CHECK-NONPOD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1
}

void test5(foo &f, const foo &x) {
  f = x;
// CHECK-POD: test5
// CHECK-POD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1

// CHECK-NONPOD: test5
// CHECK-NONPOD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 17, i1
}

extern foo globtest;

void test6(foo &&x) {
  globtest = move(x);
// CHECK-POD: test6
// CHECK-POD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1

// CHECK-NONPOD: test6
// CHECK-NONPOD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 17, i1
}

void byval(foo f);

void test7(const foo &x) {
  byval(x);
// CHECK-POD: test7
// CHECK-POD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1

// CHECK-NONPOD: test7
// CHECK-NONPOD: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} align 8 {{.*}} align 8 {{.*}} i64 24, i1
}
