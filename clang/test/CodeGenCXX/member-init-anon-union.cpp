// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - | FileCheck %s

// PR10531.

int make_a();

static union {
  int a = make_a();
  char *b;
};

int f() { return a; }

// CHECK: define internal void @__cxx_global_var_init
// CHECK-NOT: }
// CHECK: call {{.*}}@"[[CONSTRUCT_GLOBAL:.*]]C1Ev"


int g() {
  union {
    int a;
    int b = 81;
  };
  // CHECK: define {{.*}}_Z1gv
  // CHECK-NOT: }
  // CHECK: call {{.*}}@"[[CONSTRUCT_LOCAL:.*]]C1Ev"
  return b;
}


// CHECK: define {{.*}}@"[[CONSTRUCT_LOCAL]]C2Ev"
// CHECK-NOT: }
// CHECK: store i32 81

// CHECK: define {{.*}}@"[[CONSTRUCT_GLOBAL]]C2Ev"
// CHECK-NOT: }
// CHECK: call {{.*}}@_Z6make_a
