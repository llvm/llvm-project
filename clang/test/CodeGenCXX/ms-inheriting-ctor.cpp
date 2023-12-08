// RUN: %clang_cc1 -fcxx-exceptions -triple=x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s

class F {
public:
  F(wchar_t *);
};
using a = F;
struct A {};
struct b {
  b(a, F, A);
};
template <typename, typename> struct c : b {
  c(const a &p1, const A &d) : b(p1, 0, d) {}
};
template <typename e> struct B : c<e, b> {
  using c<e, b>::c;
};
class f {
public:
  f(...);
}

typedef g;
class C {
public:
  C(g, f);
};
static wchar_t h;
class D {
public:
  static C E();
};

C D::E() {
  C i(B<bool>(&h, {}), f());
  return i;
}

// Inheriting ctor has internal linkage without comdat.

// CHECK-LABEL: define internal noundef ptr @"??0?$B@_N@@QEAA@AEBVF@@AEBUA@@@Z"
// CHECK-NOT:comdat
// CHECK-SAME: {{\{$}}

// non-inheriting ctro should has linkonce_odr with comdat attribute.

// CHECK-LABEL: define linkonce_odr dso_local noundef ptr @"??0?$c@_NUb@@@@QEAA@AEBVF@@AEBUA@@@Z"
// CHECK:comdat
// CHECK-SAME: {{\{$}}
