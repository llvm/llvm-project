// RUN: %clang_cc1 -std=c++2a -fconcepts-ts -x c++ %s -verify

template <typename... Args> requires ((sizeof(Args) == 1), ...)
// expected-note@-1 {{because '(sizeof(int) == 1) , (sizeof(char) == 1) , (sizeof(int) == 1)' evaluated to false}}
void f1(Args&&... args) { }
// expected-note@-1 {{candidate template ignored: constraints not satisfied [with Args = <int, char, int>]}}

using f11 = decltype(f1('a'));
using f12 = decltype(f1(1, 'b'));
using f13 = decltype(f1(1, 'b', 2));
// expected-error@-1 {{no matching function for call to 'f1'}}

template <typename... Args>
void f2(Args&&... args) requires ((sizeof(args) == 1), ...) { }
// expected-note@-1 {{candidate template ignored: constraints not satisfied [with Args = <int, char, int>]}}
// expected-note@-2 {{because '(sizeof (args) == 1) , (sizeof (args) == 1) , (sizeof (args) == 1)' evaluated to false}}

using f21 = decltype(f2('a'));
using f22 = decltype(f2(1, 'b'));
using f23 = decltype(f2(1, 'b', 2));
// expected-error@-1 {{no matching function for call to 'f2'}}

template <typename... Args> requires ((sizeof(Args) == 1), ...)
// expected-note@-1 {{because '(sizeof(int) == 1) , (sizeof(char) == 1) , (sizeof(int) == 1)' evaluated to false}}
void f3(Args&&... args) requires ((sizeof(args) == 1), ...) { }
// expected-note@-1 {{candidate template ignored: constraints not satisfied [with Args = <int, char, int>]}}

using f31 = decltype(f3('a'));
using f32 = decltype(f3(1, 'b'));
using f33 = decltype(f3(1, 'b', 2));
// expected-error@-1 {{no matching function for call to 'f3'}}
