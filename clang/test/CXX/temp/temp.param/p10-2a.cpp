// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

template<typename T>
concept C1 = sizeof(T) == 1;
// expected-note@-1 2{{because 'sizeof(short) == 1' (2 == 1) evaluated to false}}
// expected-note@-2 {{because 'sizeof(int) == 1' (4 == 1) evaluated to false}}

template<C1 T> // expected-note {{because 'int' does not satisfy 'C1'}}
using A = T;

using a1 = A<int>; // expected-error {{constraints not satisfied for alias template 'A' [with T = int]}}
using a2 = A<char>;

template<typename T>
concept C2 = sizeof(T) == 2;
// expected-note@-1 {{because 'sizeof(char) == 2' (1 == 2) evaluated to false}}

template<C1 T1, C2 T2>
// expected-note@-1 2{{because 'short' does not satisfy 'C1'}}
// expected-note@-2 {{because 'char' does not satisfy 'C2'}}
using B = T1;

using b1 = B<char, short>;
using b2 = B<char, char>; // expected-error {{constraints not satisfied for alias template 'B' [with T1 = char, T2 = char]}}
using b3 = B<short, short>; // expected-error {{constraints not satisfied for alias template 'B' [with T1 = short, T2 = short]}}
using b4 = B<short, char>; // expected-error {{constraints not satisfied for alias template 'B' [with T1 = short, T2 = char]}}

template<typename... T>
concept C3 = (sizeof(T) + ...) == 12;
// expected-note@-1 {{because 'sizeof(char [11]) == 12' (11 == 12) evaluated to false}}
// expected-note@-2 {{because 'sizeof(char [10]) == 12' (10 == 12) evaluated to false}}
// expected-note@-3 3{{because 'sizeof(int) == 12' (4 == 12) evaluated to false}}
// expected-note@-4 6{{because 'sizeof(short) == 12' (2 == 12) evaluated to false}}

template<C3 T1, C3 T2, C3 T3>
// expected-note@-1 {{because 'char [11]' does not satisfy 'C3'}}
// expected-note@-2 {{because 'char [10]' does not satisfy 'C3'}}
using C = T2;

using c1 = C<char[12], int[3], short[6]>;
using c2 = C<char[12], char[11], char[10]>;
// expected-error@-1 {{constraints not satisfied for alias template 'C' [with T1 = char [12], T2 = char [11], T3 = char [10]]}}
using c3 = C<char[12], char[12], char[10]>;
// expected-error@-1 {{constraints not satisfied for alias template 'C' [with T1 = char [12], T2 = char [12], T3 = char [10]]}}

template<C3... Ts>
// expected-note@-1 {{because 'int' does not satisfy 'C3'}}
// expected-note@-2 2{{and 'int' does not satisfy 'C3'}}
// expected-note@-3 {{because 'short' does not satisfy 'C3'}}
// expected-note@-4 5{{and 'short' does not satisfy 'C3'}}
using D = int;

using d1 = D<char[12], int[3], short[6]>;
using d2 = D<int, int, int>;
// expected-error@-1 {{constraints not satisfied for alias template 'D' [with Ts = <int, int, int>}}
using d3 = D<short, short, short, short, short, short>;
// expected-error@-1 {{constraints not satisfied for alias template 'D' [with Ts = <short, short, short, short, short, short>}}

template<typename T>
concept C4 = sizeof(T) == 4;
// expected-note@-1 3{{because 'sizeof(char) == 4' (1 == 4) evaluated to false}}

template<C4... Ts>
// expected-note@-1 2{{because 'char' does not satisfy 'C4'}}
// expected-note@-2 {{and 'char' does not satisfy 'C4'}}
using E = int;

using e1 = E<int>;
using e2 = E<char, int>; // expected-error {{constraints not satisfied for alias template 'E' [with Ts = <char, int>]}}
using e3 = E<char, char>; // expected-error {{constraints not satisfied for alias template 'E' [with Ts = <char, char>]}}
using e4 = E<>;

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

template<typename T, typename U>
concept Same = is_same_v<T, U>; // expected-note {{because 'is_same_v<long, int>' evaluated to false}}

template<Same<int> T> // expected-note {{because 'Same<long, int>' evaluated to false}}
using F = T;

using f1 = F<int>;
using f2 = F<long>; // expected-error {{constraints not satisfied for alias template 'F' [with T = long]}}

template<typename T, typename... Ts>
concept OneOf = (is_same_v<T, Ts> || ...);
// expected-note@-1 2{{because 'is_same_v<char, char [1]>' evaluated to false}}
// expected-note@-2 2{{and 'is_same_v<char, char [2]>' evaluated to false}}
// expected-note@-3 {{because 'is_same_v<short, int>' evaluated to false}}
// expected-note@-4 {{and 'is_same_v<short, long>' evaluated to false}}
// expected-note@-5 {{and 'is_same_v<short, char>' evaluated to false}}
// expected-note@-6 3{{because 'is_same_v<int, char [1]>' evaluated to false}}
// expected-note@-7 3{{and 'is_same_v<int, char [2]>' evaluated to false}}
// expected-note@-8 2{{because 'is_same_v<nullptr_t, char>' evaluated to false}}
// expected-note@-9 2{{and 'is_same_v<nullptr_t, int>' evaluated to false}}

template<OneOf<char[1], char[2]> T, OneOf<int, long, char> U>
// expected-note@-1 2{{because 'OneOf<char, char [1], char [2]>' evaluated to false}}
// expected-note@-2 {{because 'OneOf<short, int, long, char>' evaluated to false}}
using G = T;

using g1 = G<char[1], int>;
using g2 = G<char, int>; // expected-error{{constraints not satisfied for alias template 'G' [with T = char, U = int]}}
using g3 = G<char[1], short>; // expected-error{{constraints not satisfied for alias template 'G' [with T = char [1], U = short]}}
using g4 = G<char, short>; // expected-error{{constraints not satisfied for alias template 'G' [with T = char, U = short]}}

template<OneOf<char[1], char[2]>... Ts>
// expected-note@-1 2{{because 'OneOf<int, char [1], char [2]>' evaluated to false}}
// expected-note@-2 {{and 'OneOf<int, char [1], char [2]>' evaluated to false}}
using H = int;

using h1 = H<char[1], int>;
// expected-error@-1 {{constraints not satisfied for alias template 'H' [with Ts = <char [1], int>]}}
using h2 = H<int, int>;
// expected-error@-1 {{constraints not satisfied for alias template 'H' [with Ts = <int, int>]}}
using h3 = H<char[1], char[2]>;

template<OneOf<char, int> auto x>
// expected-note@-1 {{because 'OneOf<decltype(nullptr), char, int>' evaluated to false}}
using I = int;

using i1 = I<1>;
using i2 = I<'a'>;
using i3 = I<nullptr>;
// expected-error@-1 {{constraints not satisfied for alias template 'I' [with x = nullptr]}}

template<OneOf<char, int> auto... x>
// expected-note@-1 {{because 'OneOf<decltype(nullptr), char, int>' evaluated to false}}
using J = int;

using j1 = J<1, 'b'>;
using j2 = J<'a', nullptr>;
// expected-error@-1 {{constraints not satisfied for alias template 'J' [with x = <'a', nullptr>]}}

template<OneOf<char, int> auto &x>
// expected-error@-1 {{constrained placeholder types other than simple 'auto' on non-type template parameters not supported yet}}
using K = int;
