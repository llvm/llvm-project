// RUN: %clang_cc1 -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -fsyntax-only -fdeclspec %s
int a;
void func_a(void * ptr, int a);
void func_b(int a);
void __attribute__((overloadable)) ambigious_func(void *); // expected-note {{candidate function}}
void __attribute__((overloadable)) ambigious_func(void *, int); // expected-note {{candidate function}}

inline __attribute__((noreturn(a))) void *f1(void);  // expected-error {{'noreturn' attribute takes no arguments}}
inline __attribute__((always_inline(a))) void *f2(void);  // expected-error {{'always_inline' attribute takes no arguments}}
inline __attribute__((cdecl(a))) void *f3(void);  // expected-error {{'cdecl' attribute takes no arguments}}
inline __attribute__((const(a))) void *f4(void);  // expected-error {{'const' attribute takes no arguments}}
inline __attribute__((fastcall(a))) void *f5(void);  // expected-error {{'fastcall' attribute takes no arguments}}
inline __declspec(restrict(a)) void *f6_a(void);  // expected-error {{'restrict' attribute takes no arguments}}
inline __attribute__((malloc(func_a, 1, a))) void *f6_b(void);  // expected-error {{'malloc' attribute takes no more than 2 arguments}}
inline __attribute__((malloc(func_a, 1))) void *f6_c(void);  // expected-warning {{'malloc' attribute ignored because Clang does not yet support this attribute signature}}
inline __attribute__((malloc(1234))) void *f6_d(void);  // expected-error {{'malloc' argument for deallocator is not a function}}
inline __attribute__((malloc(a))) void *f6_e(void);  // expected-error {{'malloc' argument 'a' is not a function}}
inline __attribute__((malloc(ambigious_func))) void *f6_f(void);  // expected-error {{'malloc' argument 'ambigious_func' is not a single function}}
inline __attribute__((malloc(func_b))) void *f6_g(void);  // expected-error {{'malloc' argument 'func_b' must take a pointer type as its first argument}}
inline __attribute__((malloc(func_a, 3))) void *f6_h(void);  // expected-error {{'malloc' attribute parameter 2 is out of bounds}}
inline __attribute__((malloc(func_a, 2))) void *f6_i(void);  // expected-error {{'malloc' argument '2' refers to non-pointer type 'int' of 'func_a'}}
inline __attribute__((nothrow(a))) void *f7(void);  // expected-error {{'nothrow' attribute takes no arguments}}
inline __attribute__((stdcall(a))) void *f8(void);  // expected-error {{'stdcall' attribute takes no arguments}}
inline __attribute__((used(a))) void *f9(void);  // expected-error {{'used' attribute takes no arguments}}
inline __attribute__((unused(a))) void *f10(void);  // expected-error {{'unused' attribute takes no arguments}}
inline __attribute__((weak(a))) void *f11(void);  // expected-error {{'weak' attribute takes no arguments}}

__attribute__ ((__format_arg__(2))) // expected-error {{'__format_arg__' attribute parameter 1 is out of bounds}}
void test (int, ...);

void __attribute__ ((alloc_size (2, 3))) *test2(int, ...); // expected-error {{'alloc_size' attribute parameter 1 is out of bounds}}

void gh159080_a(void);
void *gh159080_b(void) __attribute__((malloc(gh159080_a))); // expected-error{{'malloc' argument 'gh159080_a' must take a pointer type as its first argument}}
void gh159080_c();
void *gh159080_d(void) __attribute__((malloc(gh159080_c))); // expected-error{{'malloc' argument 'gh159080_c' must take a pointer type as its first argument}}

