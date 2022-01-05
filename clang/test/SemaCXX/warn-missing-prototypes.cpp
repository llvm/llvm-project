// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-prototypes -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-prototypes -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void f() { } // expected-warning {{no previous prototype for function 'f'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:1}:"static "

namespace NS {
  void f() { } // expected-warning {{no previous prototype for function 'f'}}
  // expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
}

namespace {
  // Don't warn about functions in anonymous namespaces.
  void f() { }
  // Even if they're in nested namespaces within an anonymous namespace.
  namespace NS {
    void f() { }
  }
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

// Don't warn on explicitly deleted functions.
void j() = delete;

extern void k() {} // expected-warning {{no previous prototype for function 'k'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{.*}}-[[@LINE-2]]:{{.*}}}:"{{.*}}"
