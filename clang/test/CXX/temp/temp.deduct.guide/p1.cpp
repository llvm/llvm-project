// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -verify %s -DCLASS

#ifdef CLASS
struct Outer {
#endif

template<typename> struct A {};

// Valid forms.
A(int(&)[1]) -> A<int>;
explicit A(int(&)[2]) -> A<int>;

// Declarator pieces are not OK.
*A(int(&)[3]) -> A<int>; // expected-error {{cannot specify any part of a return type in the declaration of a deduction guide}}
&A(int(&)[4]) -> A<int>; // expected-error {{cannot specify any part of a return type in the declaration of a deduction guide}}
A(int(&)[5])[3] -> A<int>;
#ifdef CLASS // FIXME: These diagnostics are both pretty bad.
// expected-error@-2 {{function cannot return array type}} expected-error@-2 {{';'}}
#else
// expected-error@-4 {{expected function body after function declarator}}
#endif

(A[3])(int(&)[5][1]) -> A<int>; // expected-error {{'<deduction guide for A>' cannot be the name of a variable}}
#ifndef CLASS
// expected-error@-2 {{declared as array of functions}}
#endif
(*A)(int(&)[5][2]) -> A<int>; // expected-error {{'<deduction guide for A>' cannot be the name of a variable}}
(&A)(int(&)[5][3]) -> A<int>; // expected-error {{'<deduction guide for A>' cannot be the name of a variable}}
(*A(int))(int(&)[5][4]) -> A<int>; // expected-error {{cannot specify any part of a return type in the declaration of a deduction guide}}

// (Pending DR) attributes and parens around the declarator-id are OK.
[[deprecated]] A(int(&)[6]) [[]] -> A<int> [[]];
A [[]] (int(&)[7]) -> A<int>;
(A)(int(&)[8]) -> A<int>;

// ... but the trailing-return-type is part of the function-declarator as normal
(A(int(&)[9])) -> A<int>;
#ifdef CLASS // FIXME: These diagnostics are both pretty bad.
// expected-error@-2 {{deduction guide declaration without trailing return type}} expected-error@-2 {{';'}}
#else
// expected-error@-4 {{expected function body after function declarator}}
#endif
(A(int(&)[10]) -> A<int>); // expected-error {{trailing return type may not be nested within parentheses}}

// A trailing-return-type is mandatory.
A(int(&)[11]); // expected-error {{deduction guide declaration without trailing return type}}

// No type specifier is permitted; we don't even parse such cases as a deduction-guide.
int A(int) -> A<int>; // expected-error {{function with trailing return type must specify return type 'auto', not 'int'}}
template<typename T> struct B {}; // expected-note {{here}}
auto B(int) -> B<int>; // expected-error {{redefinition of 'B' as different kind of symbol}}

// No storage class specifier, function specifier, ...
friend A(int(&)[20]) -> A<int>;
#ifdef CLASS
// expected-error@-2 {{cannot declare a deduction guide as a friend}}
#else
// expected-error@-4 {{'friend' used outside of class}}
#endif
typedef A(int(&)[21]) -> A<int>; // expected-error {{deduction guide cannot be declared 'typedef'}}
constexpr A(int(&)[22]) -> A<int>; // expected-error {{deduction guide cannot be declared 'constexpr'}}
inline A(int(&)[23]) -> A<int>; // expected-error {{deduction guide cannot be declared 'inline'}}
static A(int(&)[24]) -> A<int>; // expected-error {{deduction guide cannot be declared 'static'}}
thread_local A(int(&)[25]) -> A<int>; // expected-error {{'thread_local' is only allowed on variable declarations}}
extern A(int(&)[26]) -> A<int>;
#ifdef CLASS
// expected-error@-2 {{storage class specified for a member}}
#else
// expected-error@-4 {{deduction guide cannot be declared 'extern'}}
#endif
mutable A(int(&)[27]) -> A<int>; // expected-error-re {{{{'mutable' cannot be applied to|illegal storage class on}} function}}
virtual A(int(&)[28]) -> A<int>; // expected-error {{'virtual' can only appear on non-static member functions}}
const A(int(&)[31]) -> A<int>; // expected-error {{deduction guide cannot be declared 'const'}}

const volatile static constexpr inline A(int(&)[29]) -> A<int>; // expected-error {{deduction guide cannot be declared 'static inline constexpr const volatile'}}

A(int(&)[30]) const -> A<int>; // expected-error {{deduction guide cannot have 'const' qualifier}}

// No definition is allowed.
A(int(&)[40]) -> A<int> {} // expected-error {{deduction guide cannot have a function definition}}
A(int(&)[41]) -> A<int> = default; // expected-error {{deduction guide cannot have a function definition}} expected-error {{only special member functions may be defaulted}}
A(int(&)[42]) -> A<int> = delete; // expected-error {{deduction guide cannot have a function definition}}
A(int(&)[43]) -> A<int> try {} catch (...) {} // expected-error {{deduction guide cannot have a function definition}}

#ifdef CLASS
};
#endif

namespace ExplicitInst {
  // Explicit instantiation / specialization is not permitted.
  template<typename T> struct B {};
  template<typename T> B(T) -> B<T>;
  template<> B(int) -> B<int>; // expected-error {{deduction guide cannot be explicitly specialized}}
  extern template B(float) -> B<float>; // expected-error {{deduction guide cannot be explicitly instantiated}}
  template B(char) -> B<char>; // expected-error {{deduction guide cannot be explicitly instantiated}}

  // An attempt at partial specialization doesn't even parse as a deduction-guide.
  template<typename T> B<T*>(T*) -> B<T*>; // expected-error 1+{{}} expected-note 0+{{}}

  struct X {
    template<typename T> struct C {};
    template<typename T> C(T) -> C<T>;
    template<> C(int) -> C<int>; // expected-error {{deduction guide cannot be explicitly specialized}}
    extern template C(float) -> C<float>; // expected-error {{expected member name or ';'}}
    template C(char) -> C<char>; // expected-error {{expected '<' after 'template'}}
  };
}
