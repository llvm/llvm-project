// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++20 -emit-llvm -mllvm -clang-emit-module-local-hints -o %t/on.ll %t/main.cpp
// RUN: FileCheck %s < %t/on.ll
// RUN: FileCheck %s --check-prefix=PLAIN < %t/on.ll
// RUN: FileCheck %s --check-prefix=HDR < %t/on.ll
// RUN: FileCheck %s --check-prefix=HDRTMPL < %t/on.ll
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++20 -emit-llvm -o - %t/main.cpp | FileCheck %s --check-prefix=OFF

//--- hdr.h
inline int hdr_inline(int x) { return x * 3; }
template <class F> int hdr_call(F f) { return f(); }

//--- main.cpp
#include "hdr.h"

struct S { int m(int x) { return x - 1; } };
int main_plain(int x) { return x + 1; }
inline int main_inline(int x) { return x * 7; }
template <class T> int main_tmpl(T x) { return hdr_call([&] { return x * 2; }); }

int use(int x) {
  return main_plain(x) + main_inline(x) + main_tmpl(x) + hdr_inline(x) +
         S().m(x);
}

// Inline functions, template instantiations and lambdas defined in main.cpp.
// CHECK-DAG: define{{.*}} i32 @_Z11main_inlinei({{.*}}) #[[INL:[0-9]+]]
// CHECK-DAG: define{{.*}} i32 @_Z9main_tmplIiEiT_({{.*}}) #[[TMPL:[0-9]+]]
// CHECK-DAG: define{{.*}} i32 @_ZZ9main_tmplIiEiT_ENKUlvE_clEv({{.*}}) #[[LAMBDA:[0-9]+]]
// CHECK-DAG: define{{.*}} i32 @_ZN1S1mEi({{.*}}) #[[MEMBER:[0-9]+]]
// CHECK-DAG: attributes #[[INL]] = { {{.*}}"frontend-hint-likely-module-local"{{.*}} }
// CHECK-DAG: attributes #[[TMPL]] = { {{.*}}"frontend-hint-likely-module-local"{{.*}} }
// CHECK-DAG: attributes #[[LAMBDA]] = { {{.*}}"frontend-hint-likely-module-local"{{.*}} }
// CHECK-DAG: attributes #[[MEMBER]] = { {{.*}}"frontend-hint-likely-module-local"{{.*}} }

// Non-inline functions in main.cpp.
// PLAIN-DAG: define{{.*}} i32 @_Z10main_plaini({{.*}}) #[[PLAIN:[0-9]+]]
// PLAIN-DAG: define{{.*}} i32 @_Z3usei({{.*}}) #[[PLAIN]]
// PLAIN: attributes #[[PLAIN]] = {
// PLAIN-NOT: frontend-hint-likely-module-local
// PLAIN-SAME: }

// Inline function defined in hdr.h.
// HDR: define{{.*}} i32 @_Z10hdr_inlinei({{.*}}) #[[HDR:[0-9]+]]
// HDR: attributes #[[HDR]] = {
// HDR-NOT: frontend-hint-likely-module-local
// HDR-SAME: }

// Template defined in hdr.h, instantiated (with a main.cpp lambda) in main.cpp.
// HDRTMPL: define{{.*}} i32 @_Z8hdr_call{{.*}}) #[[HDRTMPL:[0-9]+]]
// HDRTMPL: attributes #[[HDRTMPL]] = {
// HDRTMPL-NOT: frontend-hint-likely-module-local
// HDRTMPL-SAME: }

// OFF-NOT: frontend-hint-likely-module-local
