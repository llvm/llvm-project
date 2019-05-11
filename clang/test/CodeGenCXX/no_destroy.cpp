// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-apple-macosx10.13.0 -o - | FileCheck %s --check-prefixes=CHECK,NO_EXCEPTIONS
// RUN: %clang_cc1 -fexceptions %s -emit-llvm -triple x86_64-apple-macosx10.13.0 -o - | FileCheck %s --check-prefixes=CHECK,EXCEPTIONS

struct NonTrivial {
  ~NonTrivial();
};

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK-NOT: __cxa_atexit{{.*}}_ZN10NonTrivialD1Ev
[[clang::no_destroy]] NonTrivial nt1;
// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK-NOT: _tlv_atexit{{.*}}_ZN10NonTrivialD1Ev
[[clang::no_destroy]] thread_local NonTrivial nt2;

struct NonTrivial2 {
  ~NonTrivial2();
};

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: __cxa_atexit{{.*}}_ZN11NonTrivial2D1Ev
NonTrivial2 nt21;
// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: _tlv_atexit{{.*}}_ZN11NonTrivial2D1Ev
thread_local NonTrivial2 nt22;

// CHECK-LABEL: define void @_Z1fv
void f() {
  // CHECK: __cxa_atexit{{.*}}_ZN11NonTrivial2D1Ev
  static NonTrivial2 nt21;
  // CHECK: _tlv_atexit{{.*}}_ZN11NonTrivial2D1Ev
  thread_local NonTrivial2 nt22;
}

// CHECK-LABEL: define void @_Z1gv
void g() {
  // CHECK-NOT: __cxa_atexit
  [[clang::no_destroy]] static NonTrivial2 nt21;
  // CHECK-NOT: _tlv_atexit
  [[clang::no_destroy]] thread_local NonTrivial2 nt22;
}

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: __cxa_atexit{{.*}}_ZN10NonTrivialD1Ev
[[clang::always_destroy]] NonTrivial nt3;
// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: _tlv_atexit{{.*}}_ZN10NonTrivialD1Ev
[[clang::always_destroy]] thread_local NonTrivial nt4;


struct NonTrivial3 {
  NonTrivial3();
  ~NonTrivial3();
};

[[clang::no_destroy]] NonTrivial3 arr[10];

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: {{invoke|call}} void @_ZN11NonTrivial3C1Ev
// EXCEPTIONS: call void @_ZN11NonTrivial3D1Ev
// NO_EXCEPTIONS-NOT: call void @_ZN11NonTrivial3D1Ev
// CHECK-NOT: call i32 @__cxa_atexit

void h() {
  [[clang::no_destroy]] static NonTrivial3 slarr[10];
}

// CHECK-LABEL: define void @_Z1hv
// CHECK: {{invoke|call}} void @_ZN11NonTrivial3C1Ev
// EXCEPTIONS: call void @_ZN11NonTrivial3D1Ev
// NO_EXCEPTIONS-NOT: call void @_ZN11NonTrivial3D1Ev
// CHECK-NOT: call i32 @__cxa_atexit

void i() {
  [[clang::no_destroy]] thread_local NonTrivial3 tlarr[10];
}

// CHECK-LABEL: define void @_Z1iv
// CHECK: {{invoke|call}} void @_ZN11NonTrivial3C1Ev
// EXCEPTIONS: call void @_ZN11NonTrivial3D1Ev
// NO_EXCEPTIONS-NOT: call void @_ZN11NonTrivial3D1Ev
// CHECK-NOT: _tlv_atexit
