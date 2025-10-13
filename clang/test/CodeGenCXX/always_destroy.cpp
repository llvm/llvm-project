// RUN: %clang_cc1 %s -fc++-static-destructors=none -emit-llvm -triple x86_64-apple-macosx10.13.0 -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,NO-DTORS %s
// RUN: %clang_cc1 %s -fc++-static-destructors=thread-local -emit-llvm -triple x86_64-apple-macosx10.13.0 -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,THREAD-LOCAL-DTORS %s

struct NonTrivial {
  ~NonTrivial();
};

// CHECK-NOT: __cxa_atexit{{.*}}_ZN10NonTrivialD1Ev
NonTrivial nt1;
// NO-DTORS-NOT: _tlv_atexit{{.*}}_ZN10NonTrivialD1Ev
// THREAD-LOCAL-DTORS: _tlv_atexit{{.*}}_ZN10NonTrivialD1Ev
thread_local NonTrivial nt2;

struct NonTrivial2 {
  ~NonTrivial2();
};

// CHECK: __cxa_atexit{{.*}}_ZN11NonTrivial2D1Ev
[[clang::always_destroy]] NonTrivial2 nt21;
// CHECK: _tlv_atexit{{.*}}_ZN11NonTrivial2D1Ev
[[clang::always_destroy]] thread_local NonTrivial2 nt22;

void f() {
  // CHECK: __cxa_atexit{{.*}}_ZN11NonTrivial2D1Ev
  [[clang::always_destroy]] static NonTrivial2 nt21;
  // CHECK: _tlv_atexit{{.*}}_ZN11NonTrivial2D1Ev
  [[clang::always_destroy]] thread_local NonTrivial2 nt22;
}

// CHECK-NOT: __cxa_atexit{{.*}}_ZN10NonTrivialD1Ev
[[clang::no_destroy]] NonTrivial nt3;
// CHECK-NOT: _tlv_atexit{{.*}}_ZN10NonTrivialD1Ev
[[clang::no_destroy]] thread_local NonTrivial nt4;
