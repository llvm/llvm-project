// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s

struct Q {
  // CHECK: @_ZN1Q1kE = linkonce_odr constant i32 5, comdat
  static constexpr int k = 5;
};
const int &r = Q::k;

int f();

// const does not imply internal linkage.
// CHECK: @external_inline = linkonce_odr constant i32 5, comdat
inline const int external_inline = 5;
const int &use1 = external_inline;

// static still does, though.
// CHECK: @_ZL15internal_inline = internal constant i32 5
static inline const int internal_inline = 5;
const int &use2 = internal_inline;

int a = f();
// CHECK: @b = linkonce_odr global i32 0, comdat
// CHECK: @_ZGV1b = linkonce_odr global i64 0, comdat($b)
inline int b = f();
int c = f();

template<typename T> struct X {
  static int a;
  static inline int b;
  static int c;
};
// CHECK: @_ZN1XIiE1aE = linkonce_odr global i32 10
// CHECK: @_ZN1XIiE1bE = global i32 20
// CHECK-NOT: @_ZN1XIiE1cE
template<> inline int X<int>::a = 10;
int &use3 = X<int>::a;
template<> int X<int>::b = 20;
template<> inline int X<int>::c = 30;

// CHECK-LABEL: define {{.*}}global_var_init
// CHECK: call i32 @_Z1fv

// CHECK-LABEL: define {{.*}}global_var_init
// CHECK-NOT: comdat
// CHECK-SAME: {{$}}
// CHECK: load atomic {{.*}} acquire
// CHECK: br
// CHECK: __cxa_guard_acquire(i64* @_ZGV1b)
// CHECK: br
// CHECK: call i32 @_Z1fv
// CHECK: __cxa_guard_release(i64* @_ZGV1b)

// CHECK-LABEL: define {{.*}}global_var_init
// CHECK: call i32 @_Z1fv

template<typename T> inline int d = f();
int e = d<int>;

// CHECK-LABEL: define {{.*}}global_var_init{{.*}}comdat
// CHECK: _ZGV1dIiE
// CHECK-NOT: __cxa_guard_acquire(i64* @_ZGV1b)
// CHECK: call i32 @_Z1fv
// CHECK-NOT: __cxa_guard_release(i64* @_ZGV1b)
