// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
auto check1() {
  return 1;
  return s; // expected-error {{use of undeclared identifier 's'}}
}

int test = 11; // expected-note 2 {{'test' declared here}}
auto check2() {
  return "s";
  return tes; // expected-error {{use of undeclared identifier 'tes'; did you mean 'test'?}}
              // expected-error@-1 {{deduced as 'int' here but deduced as 'const char *' in earlier}}
}

template <class A, class B> struct is_same { static constexpr bool value = false; };
template <class A> struct is_same<A,A> { static constexpr bool value = true; };

auto L1 = [] { return s; }; // expected-error {{use of undeclared identifier 's'}}
using T1 = decltype(L1());
static_assert(is_same<T1, void>::value, "Return statement should be discarded");
auto L2 = [] { return tes; }; // expected-error {{use of undeclared identifier 'tes'; did you mean 'test'?}}
using T2 = decltype(L2());
static_assert(is_same<T2, int>::value, "Return statement was corrected");

namespace BarNamespace {
namespace NestedNamespace { // expected-note {{'BarNamespace::NestedNamespace' declared here}}
typedef int type;
}
}
struct FooRecord { };
FooRecord::NestedNamespace::type x; // expected-error {{no member named 'NestedNamespace' in 'FooRecord'; did you mean 'BarNamespace::NestedNamespace'?}}

void cast_expr(int g) { +int(n)(g); } // expected-error {{undeclared identifier 'n'}}

void bind() { for (const auto& [test,_] : _test_) { }; } // expected-error {{undeclared identifier '_test_'}}

namespace NoCrash {
class S {
  void Function(int a) {
    unknown1(unknown2, Function, unknown3); // expected-error 2{{use of undeclared identifier}} \
                                               expected-error {{reference to non-static member function must be called}}
  }
};
}
