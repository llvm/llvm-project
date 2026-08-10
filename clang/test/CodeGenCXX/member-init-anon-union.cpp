// RUN: %clang_cc1 %s -std=c++11 -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

// PR10531.

int make_a();

static union {
  int a = make_a();
  char *b;
};

int f() { return a; }

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK-NOT: }
// CHECK: call {{.*}}@"[[CONSTRUCT_GLOBAL:.*]]C1Ev"


int g() {
  union {
    int a;
    int b = 81;
  };
  // CHECK-LABEL: define {{.*}}_Z1gv
  // CHECK-NOT: }
  // CHECK: call {{.*}}@"[[CONSTRUCT_LOCAL:.*]]C1Ev"
  return b;
}

struct A {
  A();
};
union B {
  int k;
  struct {
    A x;
    int y = 123;
  };
  B() {}
  B(int n) : k(n) {}
};

B b1;
B b2(0);

// CHECK: define {{.*}}@"[[CONSTRUCT_GLOBAL]]C2Ev"
// CHECK-NOT: }
// CHECK: call {{.*}}@_Z6make_a

// CHECK: define {{.*}}@"[[CONSTRUCT_LOCAL]]C2Ev"
// CHECK-NOT: }
// CHECK: store i32 81

// CHECK-LABEL: define {{.*}} @_ZN1BC2Ev(
// CHECK: call void @_ZN1AC1Ev(
// CHECK: store i32 123,
// CHECK: }

// CHECK-LABEL: define {{.*}} @_ZN1BC2Ei(
// CHECK-NOT: call void @_ZN1AC1Ev(
// CHECK-NOT: store i32 123,
// CHECK: store i32 %
// CHECK-NOT: call void @_ZN1AC1Ev(
// CHECK-NOT: store i32 123,
// CHECK: }
