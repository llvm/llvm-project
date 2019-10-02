// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++2a %s -emit-llvm -o - | FileCheck %s

// CHECK-DAG: @a = external thread_local global i32
extern thread_local int a;

// CHECK-DAG: @b = external thread_local global i32
extern thread_local constinit int b;

// CHECK-LABEL: define i32 @_Z1fv()
// CHECK: call i32* @_ZTW1a()
// CHECK: }
int f() { return a; }

// CHECK-LABEL: define linkonce_odr {{.*}} @_ZTW1a()
// CHECK: br i1
// CHECK: call void @_ZTH1a()
// CHECK: }

// CHECK-LABEL: define i32 @_Z1gv()
// CHECK-NOT: call
// CHECK: load i32, i32* @b
// CHECK-NOT: call
// CHECK: }
int g() { return b; }

// CHECK-NOT: define {{.*}} @_ZTW1b()

extern thread_local int c;

// CHECK-LABEL: define i32 @_Z1hv()
// CHECK: call i32* @_ZTW1c()
// CHECK: load i32, i32* %
// CHECK: }
int h() { return c; }

// Note: use of 'c' does not trigger initialization of 'd', because 'c' has a
// constant initializer.
// CHECK-LABEL: define weak_odr {{.*}} @_ZTW1c()
// CHECK-NOT: br i1
// CHECK-NOT: call
// CHECK: ret i32* @c
// CHECK: }

thread_local int c = 0;

int d_init();

// CHECK: define {{.*}}[[D_INIT:@__cxx_global_var_init[^(]*]](
// CHECK: call {{.*}} @_Z6d_initv()
thread_local int d = d_init();

struct Destructed {
  int n;
  ~Destructed();
};

extern thread_local constinit Destructed e;
// CHECK-LABEL: define i32 @_Z1iv()
// CHECK: call {{.*}}* @_ZTW1e()
// CHECK: }
int i() { return e.n; }

// CHECK: define {{.*}}[[E2_INIT:@__cxx_global_var_init[^(]*]](
// CHECK: call {{.*}} @__cxa_thread_atexit({{.*}} @_ZN10DestructedD1Ev {{.*}} @e2
thread_local constinit Destructed e2;

// CHECK-LABEL: define {{.*}}__tls_init
// CHECK: call {{.*}} [[D_INIT]]
// CHECK: call {{.*}} [[E2_INIT]]
