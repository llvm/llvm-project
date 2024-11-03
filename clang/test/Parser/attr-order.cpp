// RUN: %clang_cc1 -fsyntax-only -fms-extensions -Wno-ignored-attributes -verify %s

struct [[]] __attribute__((lockable)) __declspec(dllexport) A {}; // ok
struct [[]] __declspec(dllexport) __attribute__((lockable)) B {}; // ok
struct [[]] [[]] __declspec(dllexport) __attribute__((lockable)) C {}; // ok
struct __declspec(dllexport) [[]] __attribute__((lockable)) D {}; // ok
struct __declspec(dllexport) __attribute__((lockable)) [[]] E {}; // ok
struct __attribute__((lockable)) __declspec(dllexport) [[]] F {}; // ok
struct __attribute__((lockable)) [[]] __declspec(dllexport) G {}; // ok
struct [[]] __attribute__((lockable)) [[]] __declspec(dllexport) H {}; // ok

[[noreturn]] __attribute__((cdecl)) __declspec(dllexport) void a(); // ok
[[noreturn]] __declspec(dllexport) __attribute__((cdecl)) void b(); // ok
[[]] [[noreturn]] __attribute__((cdecl)) __declspec(dllexport) void c(); // ok

__declspec(dllexport) [[noreturn]] __attribute__((cdecl)) void d(); // expected-error {{an attribute list cannot appear here}}
__declspec(dllexport) __attribute__((cdecl)) [[noreturn]] void e(); // expected-error {{an attribute list cannot appear here}}
__attribute__((cdecl)) __declspec(dllexport) [[noreturn]] void f(); // expected-error {{an attribute list cannot appear here}}

__attribute__((cdecl)) [[noreturn]] __declspec(dllexport) void g(); // ok

[[noreturn]] __attribute__((cdecl))
[[]]
__declspec(dllexport) void h();

template <int a>
__attribute__((cdecl)) [[noreturn]] __declspec(dllexport) void i(); // ok

template <int a>
[[]] [[noreturn]] __attribute__((cdecl)) __declspec(dllexport) void j(); // ok

template <int a>
[[noreturn]] __declspec(dllexport) __attribute__((cdecl)) void k(); // ok
