// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-prototypes %s

void f() { } // expected-warning {{no previous prototype for function 'f'}}

namespace NS {
  void f() { } // expected-warning {{no previous prototype for function 'f'}}
}

namespace {
  // Don't warn about functions in anonymous namespaces.
  void f() { }
}

struct A {
  // Don't warn about member functions.
  void f() { }
};

// Don't warn about inline functions.
inline void g() { }

// Don't warn about function templates.
template<typename> void h() { }

// Don't warn when instantiating function templates.
template void h<int>();

// PR9519: don't warn about friend functions.
class I {
  friend void I_friend() {}
};

