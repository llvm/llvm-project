// RUN: %clang_cc1 -fsyntax-only -verify %s
namespace foo {
  namespace wibble {
    struct x { int y; };

    namespace bar {
      namespace wonka {
        struct x {
          struct y { };
        };
      }
    }
  }
}

namespace bar {
  typedef int y;

  struct incomplete; // expected-note{{forward declaration of 'bar::incomplete'}}
}
void test() {
  foo::wibble::x a;
  ::bar::y b;
  a + b; // expected-error{{invalid operands to binary expression ('foo::wibble::x' and '::bar::y' (aka 'int'))}}

  ::foo::wibble::bar::wonka::x::y c;
  c + b; // expected-error{{invalid operands to binary expression ('::foo::wibble::bar::wonka::x::y' and '::bar::y' (aka 'int'))}}

  (void)sizeof(bar::incomplete); // expected-error{{invalid application of 'sizeof' to an incomplete type 'bar::incomplete'}}
}

int ::foo::wibble::bar::wonka::x::y::* ptrmem;

