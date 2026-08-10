// RUN: %clang %s -fsyntax-only -fbracket-depth=512
// RUN: not %clang %s -fsyntax-only -fbracket-depth=512 -DFAIL 2>&1 | FileCheck %s

template <class T> int foo(T &&t);

void bar(int x = foo(

[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(
[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(

[](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo([](int x = foo(

#ifdef FAIL
[](int x = foo(
#endif

[](int x = foo(1)){}

#ifdef FAIL
)){}
#endif

)){})){})){})){})){})){}

)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
)){})){})){})){})){})){})){})){}
));

// CHECK: fatal error: function scope depth exceeded maximum of 127
