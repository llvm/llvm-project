// RUN: %clang_cc1 -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -fsyntax-only %s
int a;

inline __attribute__((noreturn(a))) void *f1(void);  // expected-error {{'noreturn' attribute takes no arguments}}
inline __attribute__((always_inline(a))) void *f2(void);  // expected-error {{'always_inline' attribute takes no arguments}}
inline __attribute__((cdecl(a))) void *f3(void);  // expected-error {{'cdecl' attribute takes no arguments}}
inline __attribute__((const(a))) void *f4(void);  // expected-error {{'const' attribute takes no arguments}}
inline __attribute__((fastcall(a))) void *f5(void);  // expected-error {{'fastcall' attribute takes no arguments}}
inline __attribute__((malloc(a))) void *f5(void);  // expected-error {{'malloc' attribute takes no arguments}}
inline __attribute__((nothrow(a))) void *f7(void);  // expected-error {{'nothrow' attribute takes no arguments}}
inline __attribute__((stdcall(a))) void *f8(void);  // expected-error {{'stdcall' attribute takes no arguments}}
inline __attribute__((used(a))) void *f9(void);  // expected-error {{'used' attribute takes no arguments}}
inline __attribute__((unused(a))) void *f10(void);  // expected-error {{'unused' attribute takes no arguments}}
inline __attribute__((weak(a))) void *f11(void);  // expected-error {{'weak' attribute takes no arguments}}
