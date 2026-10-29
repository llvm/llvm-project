// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -emit-llvm-only %s

// GH149372: redeclarations must not duplicate the versions in the resolver.

// CHECK: @_Z1ci.ifunc = weak_odr alias i32 (i32), ptr @_Z1ci
// CHECK: @_Z1fi.ifunc = weak_odr alias i32 (i32), ptr @_Z1fi
// CHECK: @_ZN2ns1gEi.ifunc = weak_odr alias i32 (i32), ptr @_ZN2ns1gEi
// CHECK: @h.ifunc = weak_odr alias i32 (i32), ptr @h
// CHECK: @_ZL1si.ifunc = internal alias i32 (i32), ptr @_ZL1si
// CHECK: @_Z1di.ifunc = weak_odr alias i32 (i32), ptr @_Z1di
// CHECK: @_Z1ci = weak_odr ifunc i32 (i32), ptr @_Z1ci.resolver
// CHECK: @_Z1fi = weak_odr ifunc i32 (i32), ptr @_Z1fi.resolver
// CHECK: @_ZN2ns1gEi = weak_odr ifunc i32 (i32), ptr @_ZN2ns1gEi.resolver
// CHECK: @h = weak_odr ifunc i32 (i32), ptr @h.resolver
// CHECK: @_ZL1si = internal ifunc i32 (i32), ptr @_ZL1si.resolver
// CHECK: @_Z1di = weak_odr ifunc i32 (i32), ptr @_Z1di.resolver

int __attribute__((target_clones("default,avx"))) c(int x) { return x; }
// CHECK: define {{.*}}i32 @_Z1ci.default.1(
// CHECK: define {{.*}}i32 @_Z1ci.avx.0(
// CHECK: define weak_odr ptr @_Z1ci.resolver() #[[ATTR_RESOLVER:[0-9]+]] comdat
// CHECK: ret ptr @_Z1ci.avx.0
// CHECK: ret ptr @_Z1ci.default.1

enum { VALUE = 3 };

extern int glob_x;

int __attribute__((target_clones("default,avx"))) f(int x) { return x + glob_x + VALUE; }

int f(int);
int __attribute__((target_clones("default,avx"))) f(int);
// CHECK: define {{.*}}i32 @_Z1fi.default.1(
// CHECK: define {{.*}}i32 @_Z1fi.avx.0(
// CHECK: define weak_odr ptr @_Z1fi.resolver() #[[ATTR_RESOLVER]] comdat
// CHECK: ret ptr @_Z1fi.avx.0
// CHECK-NOT: ret ptr @_Z1fi.avx.0
// CHECK: ret ptr @_Z1fi.default.1

namespace ns {
int __attribute__((target_clones("default,avx"))) g(int x) { return x; }
int g(int);
int __attribute__((target_clones("default,avx"))) g(int);
}
// CHECK: define {{.*}}i32 @_ZN2ns1gEi.default.1(
// CHECK: define {{.*}}i32 @_ZN2ns1gEi.avx.0(
// CHECK: define weak_odr ptr @_ZN2ns1gEi.resolver() #[[ATTR_RESOLVER]] comdat
// CHECK: ret ptr @_ZN2ns1gEi.avx.0
// CHECK-NOT: ret ptr @_ZN2ns1gEi.avx.0
// CHECK: ret ptr @_ZN2ns1gEi.default.1

extern "C" int __attribute__((target_clones("default,avx"))) h(int x) { return x; }
extern "C" int h(int);
extern "C" int __attribute__((target_clones("default,avx"))) h(int);
// CHECK: define {{.*}}i32 @h.default.1(
// CHECK: define {{.*}}i32 @h.avx.0(
// CHECK: define weak_odr ptr @h.resolver() #[[ATTR_RESOLVER]] comdat
// CHECK: ret ptr @h.avx.0
// CHECK-NOT: ret ptr @h.avx.0
// CHECK: ret ptr @h.default.1

static int __attribute__((target_clones("default,avx"))) s(int x) { return x; }
static int s(int);
static int __attribute__((target_clones("default,avx"))) s(int);
int use_s(int x) { return s(x); }
// CHECK: define {{.*}}i32 @_Z5use_si(
// CHECK: call noundef i32 @_ZL1si(
// CHECK: define internal ptr @_ZL1si.resolver() #[[ATTR_RESOLVER]] {
// CHECK: ret ptr @_ZL1si.avx.0
// CHECK-NOT: ret ptr @_ZL1si.avx.0
// CHECK: ret ptr @_ZL1si.default.1

int __attribute__((target_clones("default,avx"))) d(int);
int d(int);
int __attribute__((target_clones("default,avx"))) d(int);
int d(int x) { return x; }
int d(int);
// CHECK: define {{.*}}i32 @_Z1di.default.1(
// CHECK: define {{.*}}i32 @_Z1di.avx.0(
// CHECK: define weak_odr ptr @_Z1di.resolver() #[[ATTR_RESOLVER]] comdat
// CHECK: ret ptr @_Z1di.avx.0
// CHECK-NOT: ret ptr @_Z1di.avx.0
// CHECK: ret ptr @_Z1di.default.1

// The internal function's versions are emitted at the end of the module.
// CHECK: define internal {{.*}}i32 @_ZL1si.default.1(
// CHECK: define internal {{.*}}i32 @_ZL1si.avx.0(

// CHECK: attributes #[[ATTR_RESOLVER]] = { disable_sanitizer_instrumentation }
