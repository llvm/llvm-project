// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -verify -fms-compatibility
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -verify
// expected-no-diagnostics

typedef __typeof__(sizeof(0)) size_t;

#ifdef _MSC_VER
#ifndef _CRT_USE_BUILTIN_OFFSETOF
#error _CRT_USE_BUILTIN_OFFSETOF should be predefined in MSVC-compatible modes.
#endif
#else
#ifdef _CRT_USE_BUILTIN_OFFSETOF
#error _CRT_USE_BUILTIN_OFFSETOF should not be predefined in non-MSVC-compatible modes.
#endif
#endif

#if defined _MSC_VER && !defined _CRT_USE_BUILTIN_OFFSETOF
#ifdef __cplusplus
#define offsetof(s,m) ((::size_t)&reinterpret_cast<char const volatile&>((((s*)0)->m)))
#else
#define offsetof(s,m) ((size_t)&(((s*)0)->m))
#endif
#else
#define offsetof(s,m) __builtin_offsetof(s,m)
#endif

struct S { int a; };
_Static_assert(offsetof(S, a) == 0, "");
