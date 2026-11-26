// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fsyntax-only -verify %s

// GNU attributes in the type-id of an alias-declaration apply to the alias like
// they do for the equivalent typedef.

typedef char __attribute__((aligned(16))) *T1;
using U1 = char __attribute__((aligned(16))) *;
static_assert(alignof(T1) == 16 && alignof(U1) == 16, "");

typedef char * __attribute__((aligned(16))) T2;
using U2 = char * __attribute__((aligned(16)));
static_assert(alignof(T2) == 16 && alignof(U2) == 16, "");

typedef char (* __attribute__((aligned(16))) (*T3))[2];
using U3 = char (* __attribute__((aligned(16))) (*))[2];
static_assert(alignof(T3) == 16 && alignof(U3) == 16, "");

typedef __attribute__((deprecated("type-id"))) int TOrder __attribute__((deprecated("name"))); // expected-note {{'TOrder' has been explicitly marked deprecated here}}
using UOrder __attribute__((deprecated("name"))) = __attribute__((deprecated("type-id"))) int; // expected-note {{'UOrder' has been explicitly marked deprecated here}}
TOrder to; // expected-warning {{'TOrder' is deprecated: type-id}}
UOrder uo; // expected-warning {{'UOrder' is deprecated: type-id}}

typedef __attribute__((no_sanitize("address"))) void (*TNoSanitize)(); // expected-error {{'no_sanitize' attribute only applies to functions, Objective-C methods, and global variables}}
using UNoSanitize = __attribute__((no_sanitize("address"))) void (*)(); // expected-error {{'no_sanitize' attribute only applies to functions, Objective-C methods, and global variables}}

template <int N> using DepAligned = __attribute__((aligned(N))) char; // expected-error {{requested alignment is dependent but declaration is not dependent}}

using MissingSemi = __attribute__((aligned(4))) int // expected-error {{expected ';' after alias declaration}}
