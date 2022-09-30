// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify

namespace dr2621 { // dr2621: yes
enum class E { a };
namespace One {
using E_t = E;
using enum E_t; // typedef ok
auto v = a;
}
namespace Two {
using dr2621::E;
int E; // we see this
using enum E; // expected-error {{unknown type name E}}
}
}

namespace dr2628 { // dr2628: yes

template <bool A = false, bool B = false>
struct foo {
  constexpr foo() requires (!A && !B) = delete; // #DR2628_CTOR
  constexpr foo() requires (A || B) = delete;
};

void f() {
  foo fooable; // expected-error {{call to deleted}}
  // expected-note@#DR2628_CTOR {{marked deleted here}}
}

}
