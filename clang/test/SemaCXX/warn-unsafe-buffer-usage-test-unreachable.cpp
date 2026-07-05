// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify %s

typedef unsigned __darwin_size_t;
typedef __darwin_size_t size_t;
 #define bzero(s, n) __builtin_bzero(s, n)
void __nosan_bzero(void *dst, size_t sz) { bzero(dst, sz); } // expected-warning{{function '__builtin_bzero' is unsafe}}
