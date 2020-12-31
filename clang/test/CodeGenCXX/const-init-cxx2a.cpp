// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s -std=c++2a | FileCheck %s --implicit-check-not=cxx_global_var_init --implicit-check-not=cxa_atexit

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-pch -o %t.pch %s -std=c++2a
// RUN: %clang_cc1 -triple x86_64-linux-gnu -include-pch %t.pch -x c++ /dev/null -emit-llvm -o - -std=c++2a | FileCheck %s --implicit-check-not=cxx_global_var_init --implicit-check-not=cxa_atexit

// CHECK: @a ={{.*}} global i32 123,
int a = (delete new int, 123);

struct B {
  constexpr B() {}
  constexpr ~B() { n *= 5; }
  int n = 123;
};
// CHECK: @b ={{.*}} global {{.*}} i32 123
extern constexpr B b = B();

// CHECK: @_ZL1c = internal global {{.*}} i32 123
const B c;
int use_c() { return c.n; }

struct D {
  int n;
  constexpr ~D() {}
};
D d;
// CHECK: @d ={{.*}} global {{.*}} zeroinitializer

D d_arr[3];
// CHECK: @d_arr ={{.*}} global {{.*}} zeroinitializer

thread_local D d_tl;
// CHECK: @d_tl ={{.*}} thread_local global {{.*}} zeroinitializer

// CHECK-NOT: @llvm.global_ctors

// CHECK-LABEL: define {{.*}} @_Z1fv(
void f() {
  // CHECK-NOT: call
  // CHECK: call {{.*}}memcpy
  // CHECK-NOT: call
  // CHECK: call {{.*}}memset
  // CHECK-NOT: call
  // CHECK: }
  constexpr B b;
  D d = D();
}

// CHECK-LABEL: define {{.*}} @_Z1gv(
void g() {
  // CHECK-NOT: call
  // CHECK-NOT: cxa_guard
  // CHECK-NOT: _ZGV
  // CHECK: }
  static constexpr B b1;
  static const B b2;
  static D d;
  thread_local D d_tl;
}
