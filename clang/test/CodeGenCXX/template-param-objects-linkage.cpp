// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 %s -emit-llvm -o - | FileCheck %s

struct S { char buf[32]; };
template<S s> constexpr const char* f() { return s.buf; }
const char* fbuf = f<S{"a"}>();
// CHECK: @_ZTAXtl1StlA32_cLc97EEEE = linkonce_odr constant { <{ i8, [31 x i8] }> }

namespace {
  struct UN { char buf[64]; };
}
template <UN un> constexpr const char* g() { return un.buf; }
const char* gbuf = g<UN{"b"}>();
// CHECK: @_ZTAXtlN12_GLOBAL__N_12UNEtlA64_cLc98EEEE = internal constant { <{ i8, [63 x i8] }> }

struct Foo { int *i; };
int m = 0;
namespace { int n; }

template <Foo foo>
const int* h() { return foo.i; }

const int* hm = h<Foo{&m}>();
// CHECK: @_ZTAXtl3FooadL_Z1mEEE = linkonce_odr constant %struct.Foo { ptr @m }

const int* hn = h<Foo{&n}>();
// CHECK: @_ZTAXtl3FooadL_ZN12_GLOBAL__N_11nEEEE = internal constant %struct.Foo { ptr @_ZN12_GLOBAL__N_11nE }
