// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -Wno-strict-prototypes %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wno-strict-prototypes %s

int x __attribute__((constructor)); // expected-warning {{'constructor' attribute only applies to functions}}
int f(void) __attribute__((constructor));
int f(void) __attribute__((constructor(1)));
int f(void) __attribute__((constructor(1,2))); // expected-error {{'constructor' attribute takes no more than 1 argument}}
int f(void) __attribute__((constructor(1.0))); // expected-error {{'constructor' attribute requires an integer constant}}
int f(void) __attribute__((constructor(0x100000000))); // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
void knr() __attribute__((constructor));

#ifdef __cplusplus
template <float P> [[gnu::constructor(P)]] void f(); // expected-error {{'gnu::constructor' attribute requires an integer constant}}
template <double P> [[gnu::constructor(P)]] void f(); // expected-error {{'gnu::constructor' attribute requires an integer constant}}
template <int *P> [[gnu::constructor(P)]] void f(); // expected-error {{'gnu::constructor' attribute requires an integer constant}}

template <long long P> [[gnu::constructor(P)]] void f() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}} 
template void f<1LL<<32>(); // expected-note {{in instantiation of function template specialization 'f<4294967296LL>' requested here}}
template void f<101>();

template <typename T> [[gnu::constructor(static_cast<T>(1LL<<32))]] void f() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}} 
template void f<long long>(); // expected-note {{in instantiation of function template specialization 'f<long long>' requested here}}
template void f<int>();

template <typename T>
[[gnu::constructor(static_cast<T>(101))]] void g() {}
template void g<int>();
template void g<long long>();

template <typename T>
[[gnu::constructor(static_cast<T>(T{101}))]] void h() {}
template void h<int>();
template void h<long long>();

template <typename T>
[[gnu::constructor(static_cast<T>(sizeof(T[101])))]] void a() {}
template void a<int>();
template void a<long long>();
#endif

int yd __attribute__((destructor)); // expected-warning {{'destructor' attribute only applies to functions}}
int fd(void) __attribute__((destructor));
int fd(void) __attribute__((destructor(1)));
int fd(void) __attribute__((destructor(1,2))); // expected-error {{'destructor' attribute takes no more than 1 argument}}
int fd(void) __attribute__((destructor(1.0))); // expected-error {{'destructor' attribute requires an integer constant}}

#ifdef __cplusplus
template <float P> [[gnu::destructor(P)]] void fd(); // expected-error {{'gnu::destructor' attribute requires an integer constant}}
template <double P> [[gnu::destructor(P)]] void fd(); // expected-error {{'gnu::destructor' attribute requires an integer constant}}
template <int *P> [[gnu::destructor(P)]] void fd(); // expected-error {{'gnu::destructor' attribute requires an integer constant}}

template <long long P> [[gnu::destructor(P)]] void fd() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}} 
template void fd<1LL<<32>(); // expected-note {{in instantiation of function template specialization 'fd<4294967296LL>' requested here}}
template void fd<101>();

template <typename T> [[gnu::destructor(static_cast<T>(1LL<<32))]] void fd() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}} 
template void fd<long long>(); // expected-note {{in instantiation of function template specialization 'fd<long long>' requested here}}
template void fd<int>();

template <typename T>
[[gnu::destructor(static_cast<T>(101))]] void gd() {}
template void gd<int>();
template void gd<long long>();

template <typename T>
[[gnu::destructor(static_cast<T>(T{101}))]] void hd() {}
template void hd<int>();
template void hd<long long>();

template <typename T>
[[gnu::destructor(static_cast<T>(sizeof(T[101])))]] void ad() {}
template void ad<int>();
template void ad<long long>();
#endif
