// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

using T = void ();

void f1non_templ(int a) requires true; // expected-error{{non-templated function cannot have a requires clause}}
auto f2non_templ(int a) -> bool requires true; // expected-error{{non-templated function cannot have a requires clause}}
auto f3non_templ(int a) -> bool (*)(int b) requires true; // expected-error{{non-templated function cannot have a requires clause}}
// expected-error@+1{{non-templated function cannot have a requires clause}}
auto f4_non_templ(int a) requires true -> bool; // expected-error{{trailing return type must appear before trailing requires clause}}
void (f7non_templ()) requires true; // expected-error{{non-templated function cannot have a requires clause}}
// expected-error@+1{{non-templated function cannot have a requires clause}}
void (f8non_templ() requires true); // expected-error{{trailing requires clause should be placed outside parentheses}}
// expected-error@+1{{non-templated function cannot have a requires clause}}
T xnon_templ requires true;
// expected-error@+2 2{{non-templated function cannot have a requires clause}}
struct Snon_templ {
  T m1 requires true, m2 requires true;
};

template <typename>
void f1(int a)
  requires true;                               // OK

template <typename>
auto f2(int a) -> bool
  requires true;                                 // OK

template <typename>
auto f3(int a) -> bool (*)(int b) requires true; // OK
template <typename>
auto f4(int a) requires true -> bool; // expected-error{{trailing return type must appear before trailing requires clause}}
int f5(int a) requires; // expected-error{{expected expression}}
int f6(int a) requires {} // expected-error{{expected expression}}
template<typename>
void (f7()) requires true;
template<typename>
void (f8() requires true); // expected-error{{trailing requires clause should be placed outside parentheses}}
void (*(f9 requires (true)))(); // expected-error{{trailing requires clause should be placed outside parentheses}}
static_assert(is_same_v<decltype(f9), void (*)()>);
void (*pf)() requires true; // expected-error{{trailing requires clause can only be used when declaring a function}}
void g1(int (*dsdads)() requires false); // expected-error{{trailing requires clause can only be used when declaring a function}}
void g2(int (*(*dsdads)())() requires true); // expected-error{{trailing requires clause can only be used when declaring a function}}
void g3(int (*(*dsdads)(int) requires true)() ); // expected-error{{trailing requires clause should be placed outside parentheses}}

template<typename U>
struct Foo{
T x requires true;
};
Foo<int> f;

template<typename>
struct S {
  T m1 requires true, m2 requires true;
};

template<typename T>
struct R {
    R(T t);
};

template<typename T>
R(T) -> R<T> requires true;
