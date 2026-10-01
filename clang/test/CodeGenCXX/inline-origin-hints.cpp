// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++20 -emit-llvm -o - %t/main.cpp | FileCheck %s

//--- hdr.h
inline int hdr_inline(int x) { return x * 3; }
template <class F> int hdr_call(F f) { return f(); }

//--- main.cpp
#include "hdr.h"

int main_plain(int x) { return x + 1; }
inline int main_inline(int x) { return x * 7; }
template <class T> int main_tmpl(T x) { return hdr_call([&] { return x * 2; }); }

int use(int x) { return main_plain(x) + main_inline(x) + main_tmpl(x) + hdr_inline(x); }

// CHECK: define{{.*}} i32 @_Z10main_plaini({{.*}}) #[[PLAIN:[0-9]+]]
// CHECK: define{{.*}} i32 @_Z3usei({{.*}}) #[[PLAIN]]
// CHECK: define{{.*}} i32 @_Z11main_inlinei({{.*}}) #[[INL:[0-9]+]]
// CHECK: define{{.*}} i32 @_Z9main_tmplIiEiT_({{.*}}) #[[INL]]
// CHECK: define{{.*}} i32 @_Z10hdr_inlinei({{.*}}) #[[HDR:[0-9]+]]
// CHECK: define{{.*}} i32 @_ZZ9main_tmplIiEiT_ENKUlvE_clEv({{.*}}) #[[LAMBDA:[0-9]+]]

// CHECK: attributes #[[PLAIN]] = { {{.*}}"clang-main-file" {{.*}} }
// CHECK: attributes #[[INL]] = { {{.*}}"clang-main-file"="inline-or-template"{{.*}} }
// CHECK: attributes #[[HDR]] = {
// CHECK-NOT: clang-
// CHECK: attributes #[[LAMBDA]] = { {{.*}}"clang-lambda" "clang-main-file"="inline-or-template"{{.*}} }
