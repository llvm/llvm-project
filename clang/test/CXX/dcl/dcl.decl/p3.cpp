// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

void f1(int a) requires true; // OK
auto f2(int a) -> bool requires true; // OK
auto f3(int a) -> bool (*)(int b) requires true; // OK
auto f4(int a) requires true -> bool; // expected-error{{trailing return type must appear before trailing requires clause}}
int f5(int a) requires; // expected-error{{expected expression}}
int f6(int a) requires {} // expected-error{{expected expression}}
void (f7()) requires true;
void (f8() requires true); // expected-error{{trailing requires clause should be placed outside parentheses}}
void (*(f9 requires (true)))(); // expected-error{{trailing requires clause should be placed outside parentheses}}
static_assert(is_same_v<decltype(f9), void (*)()>);
void (*pf)() requires true; // expected-error{{trailing requires clause can only be used when declaring a function}}
void g1(int (*dsdads)() requires false); // expected-error{{trailing requires clause can only be used when declaring a function}}
void g2(int (*(*dsdads)())() requires true); // expected-error{{trailing requires clause can only be used when declaring a function}}
void g3(int (*(*dsdads)(int) requires true)() ); // expected-error{{trailing requires clause should be placed outside parentheses}}
using T = void ();
T x requires true;
struct S {
  T m1 requires true, m2 requires true;
};

template<typename T>
struct R {
    R(T t);
};

template<typename T>
R(T) -> R<T> requires true; // expected-error{{deduction guide cannot have a requires clause}}
