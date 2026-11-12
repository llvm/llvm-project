// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 %s -emit-llvm -o - | FileCheck %s

struct S { char buf[32]; };
template<S s> constexpr const char* f() { return s.buf; }
const char* fbuf = f<S{"a"}>();
// CHECK: @_ZTAXtl1StlA32_cLc97EEEE = linkonce_odr constant { <{ i8, [31 x i8] }> }

struct __attribute__ ((visibility ("hidden"))) HN { char buf[64]; };
template <HN hn> constexpr const char* g() { return hn.buf; }
const char* gbuf = g<HN{"b"}>();
// CHECK: @_ZTAXtl2HNtlA64_cLc98EEEE = linkonce_odr hidden constant { <{ i8, [63 x i8] }> }
