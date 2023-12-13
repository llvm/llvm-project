// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

bool r1 = requires () {};
// expected-error@-1 {{a requires expression must contain at least one requirement}}

bool r2 = requires { requires true; };

bool r3 = requires (int a, ...) { requires true; };
// expected-error@-1 {{varargs not allowed in requires expression}}

template<typename... T>
bool r4 = requires (T... ts) { requires true; };

bool r5 = requires (bool c, int d) { c; d; };

bool r6 = requires (bool c, int d) { c; d; } && decltype(d){};
// expected-error@-1 {{use of undeclared identifier 'd'}}

bool r7 = requires (bool c) { c; (requires (int d) { c; d; }); d; } && decltype(c){} && decltype(d){};
// expected-error@-1 2{{use of undeclared identifier 'd'}}
// expected-error@-2 {{use of undeclared identifier 'c'}}

bool r8 = requires (bool, int) { requires true; };

bool r9 = requires (bool a, int a) { requires true; };
// expected-error@-1 {{redefinition of parameter 'a'}}
// expected-note@-2 {{previous declaration is here}}

bool r10 = requires (struct new_struct { int x; } s) { requires true; };
// expected-error@-1 {{'new_struct' cannot be defined in a parameter type}}

bool r11 = requires (int x(1)) { requires true; };
// expected-error@-1 {{expected parameter declarator}}
// expected-error@-2 {{expected ')'}}
// expected-note@-3 {{to match this '('}}

bool r12 = requires (int x = 10) { requires true; };
// expected-error@-1 {{default arguments not allowed for parameters of a requires expression}}

bool r13 = requires (int f(int)) { requires true; };

bool r14 = requires (int (*f)(int)) { requires true; };

bool r15 = requires (10) { requires true; };
// expected-error@-1 {{expected parameter declarator}}
// expected-error@-2 {{expected ')'}} expected-note@-2 {{to match}}

bool r16 = requires (auto x) { requires true; };
// expected-error@-1 {{'auto' not allowed in requires expression parameter}}

bool r17 = requires (auto [x, y]) { requires true; };
// expected-error@-1 {{'auto' not allowed in requires expression parameter}}
// expected-error@-2 {{use of undeclared identifier 'x'}}

using a = int;

bool r18 = requires { typename a; };

bool r19 = requires { typename ::a; };

template<typename T> struct identity { using type = T; };

template<typename T> using identity_t = T;

bool r20 = requires {
    typename identity<int>::type;
    typename identity<int>;
    typename ::identity_t<int>;
};

struct s { bool operator==(const s&); ~s(); };

bool r21 = requires { typename s::operator==; };
// expected-error@-1 {{expected an identifier or template-id after '::'}}

bool r22 = requires { typename s::~s; };
// expected-error@-1 {{expected an identifier or template-id after '::'}}

template<typename T>
bool r23 = requires { typename identity<T>::temp<T>; };
// expected-error@-1 {{use 'template' keyword to treat 'temp' as a dependent template name}}

template<typename T>
bool r24 = requires {
    typename identity<T>::template temp<T>;
    typename identity<T>::template temp; // expected-error{{expected an identifier or template-id after '::'}}
};

bool r25 = requires { ; };
// expected-error@-1 {{expected expression}}

bool r26 = requires { {}; };
// expected-error@-1 {{expected expression}}

bool r27 = requires { { 0 } noexcept; };

bool r28 = requires { { 0 } noexcept noexcept; };
// expected-error@-1 {{expected '->' before expression type requirement}}
// expected-error@-2 {{expected concept name with optional arguments}}

template<typename T>
concept C1 = true;

template<typename T, typename U>
concept C2 = true;

bool r29 = requires { { 0 } noexcept C1; };
// expected-error@-1 {{expected '->' before expression type requirement}}

bool r30 = requires { { 0 } noexcept -> C2<int>; };

namespace ns { template<typename T> concept C = true; }

bool r31 = requires { { 0 } noexcept -> ns::C; };

template<typename T>
T i1 = 0;

bool r32 = requires { requires false, 1; };
// expected-error@-1 {{expected ';' at end of requirement}}

bool r33 = requires { 0 noexcept; };
// expected-error@-1 {{'noexcept' can only be used in a compound requirement (with '{' '}' around the expression)}}

bool r34 = requires { 0 int; };
// expected-error@-1 {{expected ';' at end of requirement}}

bool r35 = requires { requires true };
// expected-error@-1 {{expected ';' at end of requirement}}

bool r36 = requires (bool b) { requires sizeof(b) == 1; };

template<typename T>
void r37(bool b) requires requires { 1 } {}
// expected-error@-1 {{expected ';' at end of requirement}}

bool r38 = requires { requires { 1; }; };
// expected-error@-1 {{requires expression in requirement body; did you intend to place it in a nested requirement? (add another 'requires' before the expression)}}

bool r39 = requires { requires () { 1; }; };
// expected-error@-1 {{requires expression in requirement body; did you intend to place it in a nested requirement? (add another 'requires' before the expression)}}

bool r40 = requires { requires (int i) { i; }; };
// expected-error@-1 {{requires expression in requirement body; did you intend to place it in a nested requirement? (add another 'requires' before the expression)}}

bool r41 = requires { requires (); };
// expected-error@-1 {{expected expression}}

bool r42 = requires { typename long; }; // expected-error {{expected a qualified name after 'typename'}}

template <int N>
requires requires {
 typename _BitInt(N); // expected-error {{expected a qualified name after 'typename'}}
} using r43 = void;

template <int N>
using BitInt = _BitInt(N);

template <int N>
requires requires {
 typename BitInt<N>; // ok
} using r44 = void;

namespace GH73112 {
void f() {
    requires { requires(int; } // expected-error {{expected ')'}} \
                               // expected-error {{expected expression}} \
                               // expected-note {{to match this '('}}
}
}
