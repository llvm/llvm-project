// RUN: %clang_cc1 -fsyntax-only -verify -Wno-strict-prototypes %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wno-strict-prototypes %s

int x __attribute__((constructor)); // expected-warning {{'constructor' attribute only applies to functions}}
int f(void) __attribute__((constructor));
int f(void) __attribute__((constructor(1)));
int f(void) __attribute__((constructor(1,2))); // expected-error {{'constructor' attribute takes no more than 1 argument}}
int f(void) __attribute__((constructor(1.0))); // expected-error {{'constructor' attribute requires an integer constant}}
int f(void) __attribute__((constructor(0x100000000))); // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
void knr() __attribute__((constructor));

#ifdef __cplusplus
template <int *P> 
[[gnu::constructor(P)]] void f(); // expected-error {{'gnu::constructor' attribute requires an integer constant}}

template <long long P> 
[[gnu::constructor(P)]] void f() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}} 
template void f<1LL<<32>(); // expected-note {{in instantiation of function template specialization 'f<4294967296LL>' requested here}}
template void f<101>();

template <typename T>
[[gnu::constructor(static_cast<T>(101))]] void f() {}
template void f<int>();
template void f<long long>();

template <typename T>
[[gnu::constructor(static_cast<T>(1LL<<32))]] void g() {}
template void g<int>();

template <typename T>
[[gnu::constructor(static_cast<T>(1LL<<32))]] void h() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}} 
template void h<long long>(); // expected-note {{in instantiation of function template specialization 'h<long long>' requested here}}
#endif

int y __attribute__((destructor)); // expected-warning {{'destructor' attribute only applies to functions}}
int f(void) __attribute__((destructor));
int f(void) __attribute__((destructor(1)));
int f(void) __attribute__((destructor(1,2))); // expected-error {{'destructor' attribute takes no more than 1 argument}}
int f(void) __attribute__((destructor(1.0))); // expected-error {{'destructor' attribute requires an integer constant}}

#ifdef __cplusplus
template <int *I> 
[[gnu::destructor(I)]] void f(); // expected-error {{'gnu::destructor' attribute requires an integer constant}}

template <long long P> 
[[gnu::destructor(P)]] void fd() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}} 
template void fd<1LL<<32>(); // expected-note {{in instantiation of function template specialization 'fd<4294967296LL>' requested here}}
template void fd<101>();

template <typename T>
[[gnu::destructor(static_cast<T>(101))]] void fd() {}
template void fd<int>();
template void fd<long long>();

template <typename T>
[[gnu::destructor(static_cast<T>(1LL<<32))]] void gd() {}
template void gd<int>();

template <typename T>
[[gnu::destructor(static_cast<T>(1LL<<32))]] void hd() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}} 
template void hd<long long>(); // expected-note {{in instantiation of function template specialization 'hd<long long>' requested here}}
#endif
