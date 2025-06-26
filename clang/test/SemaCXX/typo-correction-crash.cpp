// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
auto check1() {
  return 1;
  return s; // expected-error {{use of undeclared identifier 's'}}
}

int test = 11; // expected-note 3 {{'test' declared here}}
auto check2() {
  return "s";
  return tes; // expected-error {{use of undeclared identifier 'tes'}}
              // expected-error@-1 {{deduced as 'int' here but deduced as 'const char *' in earlier}}
}

template <class A, class B> struct is_same { static constexpr bool value = false; };
template <class A> struct is_same<A,A> { static constexpr bool value = true; };

auto L1 = [] { return s; }; // expected-error {{use of undeclared identifier 's'}}
using T1 = decltype(L1());
static_assert(is_same<T1, void>::value, "Return statement should be discarded");
auto L2 = [] { return tes; }; // expected-error {{use of undeclared identifier 'tes'}}
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

void bind() { for (const auto& [test,_] : _test_) { }; } // expected-error {{undeclared identifier '_test_'}} \
                                                            expected-error {{invalid range expression of type 'int'; no viable 'begin' function available}}

namespace NoCrash {
class S {
  void Function(int a) {
    unknown1(unknown2, Function, unknown3); // expected-error 2{{use of undeclared identifier}}
  }
};
}

namespace NoCrashOnCheckArgAlignment {
template <typename a> void b(a &);
void test() {
  for (auto file_data :b(files_db_data)); // expected-error {{use of undeclared identifier 'files_db_data'}}
}
}
