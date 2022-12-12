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

namespace dr2628 { // dr2628: yes open

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

namespace dr2635 { // dr2635: 16
template<typename T>
concept UnaryC = true;
template<typename T, typename U>
concept BinaryC = true;

struct S{ int i, j; };
S get_S();

template<typename T>
T get_T();

void use() {
  // expected-error@+1{{decomposition declaration cannot be declared with constrained 'auto'}}
  UnaryC auto [a, b] = get_S();
  // expected-error@+1{{decomposition declaration cannot be declared with constrained 'auto'}}
  BinaryC<int> auto [c, d] = get_S();
}

template<typename T>
void TemplUse() {
  // expected-error@+1{{decomposition declaration cannot be declared with constrained 'auto'}}
  UnaryC auto [a, b] = get_T<T>();
  // expected-error@+1{{decomposition declaration cannot be declared with constrained 'auto'}}
  BinaryC<T> auto [c, d] = get_T<T>();
}
}

  // dr2636: na

  // dr2642: na

namespace dr2644 { // dr2644: yes

auto z = [a = 42](int a) { // expected-error {{a lambda parameter cannot shadow an explicitly captured entity}} \
                           // expected-note {{variable 'a' is explicitly captured here}}
     return 1;
};

}

namespace dr2650 { // dr2650: yes
template <class T, T> struct S {};
template <class T> int f(S<T, T{}>*); // expected-note {{type 'X' of non-type template parameter is not a structural type}}
class X {
  int m;
};
int i0 = f<X>(0);   //expected-error {{no matching function for call to 'f'}}
}

namespace dr2654 { // dr2654: 16
void f() {
    int neck, tail;
    volatile int brachiosaur;
    brachiosaur += neck;                // OK
    brachiosaur -= neck;                // OK
    brachiosaur |= neck;                // OK
}
}
