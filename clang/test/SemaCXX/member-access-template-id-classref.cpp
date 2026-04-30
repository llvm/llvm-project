// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++17
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++20
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23

// Regression test: when `using namespace std;` brings the `std::hash` class
// template into the global namespace, a member access of the form
// `obj.hash < expr` (where `hash` is a non-template member of obj's class
// type) must not be disambiguated as the start of a template-id.
//
// Per [basic.lookup.classref], the identifier `hash` after `.` or `->` is
// looked up first in the class of the object expression. Only if it is not
// found there is the surrounding scope consulted. Bug reported via
// https://github.com/stephenberry/glaze/issues/2534

// expected-no-diagnostics

namespace std {
  template <class T> struct hash {};
}

using namespace std;

// Case 1: nested class inside a class template, accessed via `->` and `.`.
template <class T>
struct M1 {
  struct E { unsigned hash; };
  static const E *f(const E *p, unsigned t) {
    if (p->hash < t) return p;
    return nullptr;
  }
  static int g(E e, unsigned t) {
    return (e.hash < t) ? 1 : 0;
  }
};

// Case 2: nested class whose `hash` member type genuinely depends on T.
template <class T>
struct M2 {
  struct E { T hash; };
  static int g(E e, T t) {
    return (e.hash < t) ? 1 : 0;
  }
};

// Case 3: non-template, non-nested class. Sanity baseline (already worked
// before the fix).
struct E3 { unsigned hash; };
static int f3(const E3 *p, unsigned t) {
  if (p->hash < t) return 1;
  return 0;
}

// Case 4: direct member of the current instantiation accessed via `this->`.
// Sanity baseline (already worked before the fix).
template <class T>
struct M4 {
  unsigned hash;
  int g(unsigned t) const {
    return (this->hash < t) ? 1 : 0;
  }
};

// Force instantiation so any deferred lookup is exercised.
template struct M1<int>;
template struct M2<int>;
template struct M4<int>;
