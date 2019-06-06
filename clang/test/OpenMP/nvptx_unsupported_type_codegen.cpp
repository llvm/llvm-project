// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// expected-no-diagnostics

// CHECK-DAG: [[T:%.+]] = type {{.+}}, fp128,
// CHECK-DAG: [[T1:%.+]] = type {{.+}}, i128, i128,

struct T {
  char a;
  __float128 f;
  char c;
  T() : a(12), f(15) {}
  T &operator+(T &b) { f += b.a; return *this;}
};

struct T1 {
  char a;
  __int128 f;
  __int128 f1;
  char c;
  T1() : a(12), f(15) {}
  T1 &operator+(T1 &b) { f += b.a; return *this;}
};

#pragma omp declare target
T a = T();
T f = a;
// CHECK: define{{ dso_local | }}void @{{.+}}foo{{.+}}([[T]]* byval([[T]]) align {{.+}})
void foo(T a = T()) {
  return;
}
// CHECK: define{{ dso_local | }}[6 x i64] @{{.+}}bar{{.+}}()
T bar() {
// CHECK:      bitcast [[T]]* %{{.+}} to [6 x i64]*
// CHECK-NEXT: load [6 x i64], [6 x i64]* %{{.+}},
// CHECK-NEXT: ret [6 x i64]
  return T();
}
// CHECK: define{{ dso_local | }}void @{{.+}}baz{{.+}}()
void baz() {
// CHECK:      call [6 x i64] @{{.+}}bar{{.+}}()
// CHECK-NEXT: bitcast [[T]]* %{{.+}} to [6 x i64]*
// CHECK-NEXT: store [6 x i64] %{{.+}}, [6 x i64]* %{{.+}},
  T t = bar();
}
T1 a1 = T1();
T1 f1 = a1;
// CHECK: define{{ dso_local | }}void @{{.+}}foo1{{.+}}([[T1]]* byval([[T1]]) align {{.+}})
void foo1(T1 a = T1()) {
  return;
}
// CHECK: define{{ dso_local | }}[[T1]] @{{.+}}bar1{{.+}}()
T1 bar1() {
// CHECK:      load [[T1]], [[T1]]*
// CHECK-NEXT: ret [[T1]]
  return T1();
}
// CHECK: define{{ dso_local | }}void @{{.+}}baz1{{.+}}()
void baz1() {
// CHECK: call [[T1]] @{{.+}}bar1{{.+}}()
  T1 t = bar1();
}
#pragma omp end declare target
