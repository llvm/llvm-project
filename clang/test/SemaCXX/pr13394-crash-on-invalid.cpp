// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// Don't crash (PR13394).

namespace stretch_v1 {
  struct closure_t {
    const stretch_v1::ops_t* d_methods; // expected-error {{no type named 'ops_t' in namespace 'stretch_v1'}}
  };
}
namespace gatekeeper_v1 {
  namespace gatekeeper_factory_v1 {
    struct closure_t { // expected-note {{'closure_t' declared here}} expected-note {{'gatekeeper_factory_v1::closure_t' declared here}}
      gatekeeper_v1::closure_t* create(); // expected-error {{no type named 'closure_t' in namespace 'gatekeeper_v1'; did you mean simply 'closure_t'?}}
    };
  }
  // FIXME: Typo correction should remove the 'gatekeeper_v1::' name specifier
  gatekeeper_v1::closure_t *x; // expected-error {{no type named 'closure_t' in namespace 'gatekeeper_v1'; did you mean 'gatekeeper_factory_v1::closure_t'}}
}

namespace Foo {
struct Base {
  void Bar() {} // expected-note{{'Bar' declared here}}
};
}

struct Derived : public Foo::Base {
  void test() {
    Foo::Bar(); // expected-error{{no member named 'Bar' in namespace 'Foo'; did you mean simply 'Bar'?}}
  }
};
