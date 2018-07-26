// RUN: %clang_cc1 -triple aarch64-windows -ffreestanding -emit-llvm -O0 \
// RUN: -x c++ -o - %s | FileCheck %s

struct pod { int a, b, c, d, e; };

struct non_pod {
  int a;
  non_pod() {}
};

struct pod s;
struct non_pod t;

struct pod bar() { return s; }
struct non_pod foo() { return t; }
// CHECK: define {{.*}} void @{{.*}}bar{{.*}}(%struct.pod* noalias sret %agg.result)
// CHECK: define {{.*}} void @{{.*}}foo{{.*}}(%struct.non_pod* noalias %agg.result)


// Check instance methods.
struct pod2 { int x; };
struct Baz { pod2 baz(); };

int qux() { return Baz().baz().x; }
// CHECK: declare {{.*}} void @{{.*}}baz@Baz{{.*}}(%struct.Baz*, %struct.pod2*)
