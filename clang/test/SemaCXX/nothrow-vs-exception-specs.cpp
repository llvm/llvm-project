// RUN: %clang_cc1 %s -fcxx-exceptions -fdeclspec -fsyntax-only -Wexceptions -verify -std=c++14
// RUN: %clang_cc1 %s -fcxx-exceptions -fdeclspec -fsyntax-only -Wexceptions -verify -std=c++17 -DCPP17

__attribute__((nothrow)) void f1();
static_assert(noexcept(f1()), "");
void f1() noexcept;
// expected-error@+2 {{exception specification in declaration does not match previous declaration}}
// expected-note@-2 {{previous declaration is here}}
void f1() noexcept(false);

__attribute__((nothrow)) void f2();
static_assert(noexcept(f2()), "");
// expected-error@+2 {{exception specification in declaration does not match previous declaration}}
// expected-note@-3 {{previous declaration is here}}
void f2() noexcept(false);

void f3() __attribute__((nothrow));
static_assert(noexcept(f3()), "");
void f3() noexcept;
// expected-error@+2 {{exception specification in declaration does not match previous declaration}}
// expected-note@-2 {{previous declaration is here}}
void f3() noexcept(false);

// Still noexcept due to throw()
__attribute__((nothrow)) void f4() throw();
static_assert(noexcept(f4()), "");

// Still noexcept due to noexcept
__attribute__((nothrow)) void f5() noexcept;
static_assert(noexcept(f5()), "");

// Still noexcept due to noexcept(true)
__attribute__((nothrow)) void f6() noexcept(true);
static_assert(noexcept(f6()), "");

#ifndef CPP17
// Doesn't override C++ implementation.
// expected-warning@+1{{'nothrow' attribute conflicts with exception specification; attribute ignored}}
__attribute__((nothrow)) void f7() throw(int);
static_assert(!noexcept(f7()), "");
#endif

// Doesn't override C++ implementation.
// expected-warning@+1{{'nothrow' attribute conflicts with exception specification; attribute ignored}}
__attribute__((nothrow)) void f8() noexcept(false);
static_assert(!noexcept(f8()), "");

__declspec(nothrow) void foo1() noexcept;
__declspec(nothrow) void foo2() noexcept(true);
// expected-warning@+1{{'nothrow' attribute conflicts with exception specification; attribute ignored}}
__declspec(nothrow) void foo3() noexcept(false);
__declspec(nothrow) void foo4() noexcept(noexcept(foo1()));
__declspec(nothrow) void foo5() noexcept(noexcept(foo2()));
// expected-warning@+1{{'nothrow' attribute conflicts with exception specification; attribute ignored}}
__declspec(nothrow) void foo6() noexcept(noexcept(foo3()));
