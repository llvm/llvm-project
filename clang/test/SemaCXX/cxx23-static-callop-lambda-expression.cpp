// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

namespace ns1 {
  auto lstatic = []() static { return 3; };
  int (*f2)(void) = lstatic;

}

namespace ns1_1 {

  auto lstatic = []() static consteval // expected-note {{operator int (*)()' is an immediate function because its body evaluates the address of a consteval function 'operator()'}} \
                                       // expected-note {{declared here}}
  { return 3; };

  int (*f2)(void) = lstatic; // expected-error {{call to immediate function 'ns1_1::(lambda)::operator int (*)()' is not a constant expression}} \
                             // expected-note {{pointer to a consteval declaration is not a constant expression}}

}


namespace ns2 {
  auto lstatic = []() static { return 3; };
  constexpr int (*f2)(void) = lstatic;
  static_assert(lstatic() == f2());
}

namespace ns3 {
  void main() {
    static int x = 10;
    auto L = []() static { return x; };
  }
}
