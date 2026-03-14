// RUN: %clang_cc1 -std=c++2a -verify %s -pedantic

template<typename T> struct Template {};

struct Linkage1 { struct Inner {}; };
typedef struct { struct Inner {}; } Linkage2;

typedef struct {} const NoLinkage1;
auto x = [] {};
typedef decltype(x) NoLinkage2;
auto f() { return [] {}; }
typedef decltype(f()) NoLinkage3;

inline auto g() { return [] {}; }
typedef decltype(g()) VisibleNoLinkage1;
inline auto y = [] {};
typedef decltype(y) VisibleNoLinkage2;
inline auto h() { struct {} x; return x; }
typedef decltype(h()) VisibleNoLinkage3;

extern Linkage1 linkage1v;
extern Linkage1::Inner linkage1iv;
extern Linkage2 linkage2v;
extern Linkage2::Inner linkage2iv;
extern Template<Linkage1> linkaget1v;
extern Linkage1 linkage1f();
void linkage2f(Linkage2);

void use_linkage() {
  &linkage1v, &linkage1iv, &linkage2v, &linkage2iv, &linkaget1v; // expected-warning 4{{left operand of comma operator has no effect}} expected-warning {{unused}}
  linkage1f();
  linkage2f({});
}

extern NoLinkage1 no_linkage1(); // expected-error {{function 'no_linkage1' is used but not defined in this translation unit}}
extern NoLinkage2 no_linkage2(); // expected-error {{function 'no_linkage2' is used but not defined in this translation unit}}
extern NoLinkage3 no_linkage3(); // expected-error {{function 'no_linkage3' is used but not defined in this translation unit}}

void use_no_linkage() {
  no_linkage1(); // expected-note {{used here}}
  no_linkage2(); // expected-note {{used here}}
  no_linkage3(); // expected-note {{used here}}
}

extern VisibleNoLinkage1 visible_no_linkage1(); // expected-warning {{ISO C++ requires a definition}}
extern VisibleNoLinkage2 visible_no_linkage2(); // expected-warning {{ISO C++ requires a definition}}
extern VisibleNoLinkage3 visible_no_linkage3(); // expected-warning {{ISO C++ requires a definition}}

void use_visible_no_linkage() {
  visible_no_linkage1(); // expected-note {{used here}}
  visible_no_linkage2(); // expected-note {{used here}}
  visible_no_linkage3(); // expected-note {{used here}}
}

namespace {
  struct InternalLinkage {};
}
InternalLinkage internal_linkage(); // expected-error {{used but not defined}}
void use_internal_linkage() {
  internal_linkage(); // expected-note {{used here}}
}

extern inline int not_defined; // expected-error {{not defined}}
extern inline int defined_after_use;
void use_inline_vars() {
  not_defined = 1; // expected-note {{used here}}
  defined_after_use = 2;
}
inline int defined_after_use;

namespace {
  template<typename T> struct A {
    static const int n;
  };
  template<typename T> const int A<T>::n = 3;
  static_assert(A<int>::n == 3);
  int k = A<float>::n;
}
