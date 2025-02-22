// RUN: %clang_cc1 -std=c++98 -pedantic-errors -verify=expected,cxx98 %s
// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=expected,since-cxx11 %s
// RUN: %clang_cc1 -std=c++14 -pedantic-errors -verify=expected,since-cxx11 %s
// RUN: %clang_cc1 -std=c++17 -pedantic-errors -verify=expected,since-cxx11 %s
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify=expected,since-cxx11,since-cxx20 %s
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify=expected,since-cxx11,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify=expected,since-cxx11,since-cxx20,since-cxx23,since-cxx26 %s

// cxx98-no-diagnostics

namespace cwg2913 { // cwg2913: 20

#if __cplusplus >= 202002L

template<typename T>
struct R {
  R(T);
  R(T, T);
};

template<typename T>
R(T) -> R<T> requires true;

template<typename T>
R(T, T) requires true -> R<T>;
// since-cxx20-error@-1 {{expected function body after function declarator}}

#endif

} // namespace cwg2913

namespace cwg2915 { // cwg2915: 20
#if __cplusplus >= 202302L
struct A {
  void f(this void);
  // since-cxx23-error@-1 {{explicit object parameter cannot have 'void' type}}
};
#endif
} // namespace cwg2915

namespace cwg2917 { // cwg2917: 20 review 2024-07-30
#if __cplusplus >= 201103L
template <typename>
class Foo;

template<class ...>
struct C {
  struct Nested { };
};

struct S {
  template <typename>
  friend class Foo, int;
  // since-cxx11-error@-1 {{a friend declaration that befriends a template must contain exactly one type-specifier}}

  template <typename ...Ts>
  friend class C<Ts>::Nested...;
  // since-cxx11-error@-1 {{friend declaration expands pack 'Ts' that is declared it its own template parameter list}}
};
#endif
} // namespace cwg2917

#if __cplusplus > 202302L
namespace std {
  using size_t = decltype(sizeof(0));
} // namespace std
void *operator new(std::size_t, void *p) { return p; }
void* operator new[] (std::size_t, void* p) {return p; }
#endif

namespace cwg2922 { // cwg2922: 20
#if __cplusplus > 202302L
union U { int a, b; };
constexpr U nondeterministic(bool i) {
  if(i) {
    U u;
    new (&u) int(); // #cwg2922-placement-new
    return u;
  }
  return {};
}
constexpr U _ = nondeterministic(true);
// since-cxx26-error@-1 {{constexpr variable '_' must be initialized by a constant expression}}
//   since-cxx26-note@#cwg2922-placement-new {{placement new would change type of storage from 'U' to 'int'}}
//   since-cxx26-note@-3 {{in call to 'nondeterministic(true)'}}
#endif
} // namespace cwg2922
