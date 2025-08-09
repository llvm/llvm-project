// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value -std=c++20 %s

namespace GH39811 {

template<int = 0> class C {};

C (a);
C (b) = C();
C (c) {};
C (((((d)))));

template<C (e)> class X;
template<C (...f)> class Y;

void test() {
    C (g);
    C (h) = C();
    C (i) {};
    (void)g;
    (void)h;
    (void)i;
}

C* (bad1); // expected-error {{cannot form pointer to deduced class template specialization type}}
C (*bad2); // expected-error {{cannot form pointer to deduced class template specialization type}}

}

namespace GH64347 {

template<typename X, typename Y> struct A { X x; Y y;};
void test() {
   A(1, 2);
   new A(1, 2);
}

template<A a>
void f() { (void)a; }
void k() {
  // Test CTAD works for non-type template arguments.
  f<A(0, 0)>();
}

} // namespace GH64347

namespace GH123591 {


template < typename... _Types >
struct variant {
  template <int N = sizeof...(_Types)>
  variant(_Types...);
};

template <class T>
using AstNode = variant<T, T, T>;

AstNode tree(42, 43, 44);

}

namespace GH123591_2 {

template <int>
using enable_if_t = char;

template < typename... Types >
struct variant {
  template < enable_if_t<sizeof...(Types)>>
  variant();
};

template <int>
using AstNode = variant<>;
// expected-note@-1 {{couldn't infer template argument ''}} \
// expected-note@-1 2{{implicit deduction guide declared as}} \
// expected-note@-1 {{candidate function template not viable}}


AstNode tree; // expected-error {{no viable constructor or deduction guide}}

}

namespace GH127539 {

template <class...>
struct A {
    template <class... ArgTs>
    A(ArgTs...) {}
};

template <class... ArgTs>
A(ArgTs...) -> A<typename ArgTs::value_type...>;

template <class... Ts>
using AA = A<Ts..., Ts...>;

AA a{};

}

namespace GH129077 {

using size_t = decltype(sizeof(0));

struct index_type
{
  size_t value = 0;
  index_type() = default;
  constexpr index_type(size_t i) noexcept : value(i) {}
};

template <index_type... Extents>
struct extents
{
  constexpr extents(decltype(Extents)...) noexcept {}
};

template <class... Extents>
extents(Extents...) -> extents<(requires { Extents::value; } ? Extents{} : ~0ull)...>;

template <index_type... Index>
using index = extents<Index...>;

int main()
{
  extents i{0,0};
  auto j = extents<64,{}>({}, 42);

  index k{0,0};
  auto l = index<64,{}>({}, 42);

  return 0;
}

}

namespace GH129620 {

template <class... Ts>
struct A {
    constexpr A(Ts...) {}
};

template <class... Ts>
using Foo = A<Ts...>;

template <class T>
using Bar = Foo<T, T>;

Bar a{0, 0};

}

namespace GH129998 {

struct converible_to_one {
    constexpr operator int() const noexcept { return 1; }
};

template <int... Extents>
struct class_template {
    class_template() = default;
    constexpr class_template(auto&&...) noexcept {}
};

template <class... Extents>
class_template(Extents...) -> class_template<(true ? 0 : +Extents{})...>;

template <int... Extents>
using alias_template = class_template<Extents...>;

alias_template var2{converible_to_one{}, 2};

}

namespace GH136624 {
  // expected-note@+1 2{{no known conversion}}
  template<typename U> struct A {
    U t;
  };

  template<typename V> A(V) -> A<V>;

  namespace foo {
    template<class Y> using Alias = A<Y>;
  }

  // FIXME: This diagnostic is missing 'foo::Alias', as written.
  foo::Alias t = 0;
  // expected-error@-1 {{no viable conversion from 'int' to 'GH136624::A<int>' (aka 'A<int>')}}
} // namespace GH136624
