// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test1 __attribute__((visibility("hidden"))) { // expected-note{{surrounding namespace with visibility attribute starts here}}
#pragma GCC visibility pop // expected-error{{#pragma visibility pop with no matching #pragma visibility push}}
}

// GCC 4.6 accepts this, but the "hidden" leaks past the namespace end.
namespace test2 __attribute__((visibility("hidden"))) {
#pragma GCC visibility push(protected) // expected-error{{#pragma visibility push with no matching #pragma visibility pop}}
} // expected-note{{surrounding namespace with visibility attribute ends here}}

#pragma GCC visibility pop // expected-error{{#pragma visibility pop with no matching #pragma visibility push}}

// <rdar://problem/10871094>
struct A {
  #pragma GCC visibility push(protected)
  #pragma GCC visibility pop
};

void f() {
  #pragma GCC visibility push(protected)
  #pragma GCC visibility pop
}

namespace pr13662 {
#pragma GCC visibility push(hidden)
  template<class T> class __attribute__((__visibility__("default"))) foo;
  class bar { template<class T> friend class foo; };
#pragma GCC visibility pop
}
