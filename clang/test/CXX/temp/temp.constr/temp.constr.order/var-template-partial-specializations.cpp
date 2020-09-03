// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

template<typename T> requires (sizeof(T) >= 4)
// expected-note@-1{{similar constraint expressions not considered equivalent}}
bool a = false; // expected-note{{template is declared here}}

template<typename T> requires (sizeof(T) >= 4 && sizeof(T) <= 10)
// expected-note@-1{{similar constraint expression here}}
bool a<T> = true; // expected-error{{variable template partial specialization is not more specialized than the primary template}}

template<typename T>
concept C1 = sizeof(T) >= 4;

template<typename T> requires C1<T>
bool b = false;

template<typename T> requires (C1<T> && sizeof(T) <= 10)
bool b<T> = true;

template<typename T>
concept C2 = sizeof(T) > 1 && sizeof(T) <= 8;

template<typename T>
bool c = false;

template<typename T> requires C1<T>
bool c<T> = true;

template<typename T>
bool d = false;

template<typename T>
bool d<T> = true; // expected-error{{variable template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}}

template<typename T> requires C1<T>
bool e = false;

template<typename T>
bool e<T> = true; // expected-error{{variable template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}}

template<typename T>
constexpr int f = 1;

template<typename T> requires C1<T> && C2<T>
constexpr int f<T> = 2;

template<typename T> requires C1<T> || C2<T>
constexpr int f<T> = 3;

static_assert(f<unsigned> == 2);
static_assert(f<char[10]> == 3);
static_assert(f<char> == 1);



