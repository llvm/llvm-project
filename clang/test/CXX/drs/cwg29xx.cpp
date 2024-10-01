// RUN: %clang_cc1 -std=c++98 -pedantic-errors -verify=expected,cxx98 %s
// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify=expected %s

namespace cwg2913 { // cwg2913: 20 tentatively ready 2024-08-16

#if __cplusplus >= 202002L

template<typename T>
struct R {
  R(T);
  R(T, T);
};

template<typename T>
R(T) -> R<T> requires true;

template<typename T>
R(T, T) requires true -> R<T>; // expected-error {{expected function body after function declarator}}

#endif

} // namespace cwg2913

namespace cwg2915 { // cwg2915: 20 tentatively ready 2024-08-16
#if __cplusplus >= 202302L
struct A {
  void f(this void); // expected-error {{explicit object parameter cannot have 'void' type}}
};
#endif
}

namespace cwg2917 { // cwg2917: 20 review 2024-07-30
template <typename>
class Foo;

template<class ...> // cxx98-error {{variadic templates are a C++11 extension}}
struct C {
  struct Nested { };
};

struct S {
  template <typename>
  friend class Foo, int; // expected-error {{a friend declaration that befriends a template must contain exactly one type-specifier}}

  template <typename ...Ts> // cxx98-error {{variadic templates are a C++11 extension}}
  friend class C<Ts>::Nested...; // expected-error {{friend declaration expands pack 'Ts' that is declared it its own template parameter list}}
};
} // namespace cwg2917

#if __cplusplus >= 202400L

namespace std {
  using size_t = decltype(sizeof(0));
};
void *operator new(std::size_t, void *p) { return p; }
void* operator new[] (std::size_t, void* p) {return p;}


namespace cwg2922 { // cwg2922: 20 tentatively ready 2024-07-10
union U { int a, b; };
constexpr U nondeterministic(bool i) {
  if(i) {
    U u;
    new (&u) int();
    // expected-note@-1 {{placement new would change type of storage from 'U' to 'int'}}
    return u;
  }
  return {};
}
constexpr U _ = nondeterministic(true);
// expected-error@-1 {{constexpr variable '_' must be initialized by a constant expression}} \
// expected-note@-1 {{in call to 'nondeterministic(true)'}}
}
#endif
