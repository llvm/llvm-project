// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

template<typename T> requires (sizeof(T) >= 4)
// expected-note@-1{{similar constraint expressions not considered equivalen}}
class A{}; // expected-note{{template is declared here}}

template<typename T> requires (sizeof(T) >= 4 && sizeof(T) <= 10)
// expected-note@-1{{similar constraint expression here}}

class A<T>{}; // expected-error{{class template partial specialization is not more specialized than the primary template}}

template<typename T>
concept C1 = sizeof(T) >= 4;

template<typename T> requires C1<T>
class B{};

template<typename T> requires (C1<T> && sizeof(T) <= 10)
class B<T>{};

template<typename T>
concept C2 = sizeof(T) > 1 && sizeof(T) <= 8;

template<typename T>
class C{};

template<typename T> requires C1<T>
class C<T>{};

template<typename T>
class D{}; // expected-note{{previous definition is here}}

template<typename T>
class D<T>{}; // expected-error{{class template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}} expected-error{{redefinition of 'D'}}

template<typename T> requires C1<T> // expected-note{{previous template declaration is here}}
class E{};

template<typename T> // expected-error{{requires clause differs in template redeclaration}}
class E<T>{}; // expected-error{{class template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}}

template<typename T>
struct F{ enum{ value = 1 }; };

template<typename T> requires C1<T> && C2<T>
struct F<T>{ enum{ value = 2 }; };

template<typename T> requires C1<T> || C2<T>
struct F<T>{ enum{ value = 3 }; };

static_assert(F<unsigned>::value == 2);
static_assert(F<char[10]>::value == 3);
static_assert(F<char>::value == 1);

// Make sure atomic constraints subsume each other only if their parameter
// mappings are identical.

template<typename T, typename U> requires C2<T>
struct I { }; // expected-note {{template is declared here}}

template<typename T, typename U> requires C2<U>
struct I<T, U> { }; // expected-error {{class template partial specialization is not more specialized than the primary template}}

template<typename T, typename U> requires C2<T> && C2<U>
struct I<T, U> { };
