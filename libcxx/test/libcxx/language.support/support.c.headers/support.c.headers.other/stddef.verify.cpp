//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is the same test as clang/test/Headers/stddef.c, but to test the
// libc++ version of stddef.h interacts properly with the clang version.

struct astruct {
  char member;
};

ptrdiff_t p0;                                 // expected-error{{unknown type name 'ptrdiff_t'}}
size_t s0;                                    // expected-error{{unknown type name 'size_t'}}
rsize_t r0;                                   // expected-error{{unknown type name 'rsize_t'}}
wchar_t wc0;                                  // wchar_t is a keyword in C++
void* v0 = NULL;                              // expected-error{{use of undeclared identifier 'NULL'}}
nullptr_t n0;                                 // expected-error{{unknown type name 'nullptr_t'}}
static void f0(void) { unreachable(); }       // expected-error{{undeclared identifier 'unreachable'}}
max_align_t m0;                               // expected-error{{unknown type name 'max_align_t'}}
size_t o0 = offsetof(struct astruct, member); // expected-error{{unknown type name 'size_t'}}
    // expected-error@-1{{expected expression}} expected-error@-1{{use of undeclared identifier 'member'}}
wint_t wi0; // expected-error{{unknown type name 'wint_t'}}

#include <stddef.h>

ptrdiff_t p1;
size_t s1;
rsize_t r1;
#if __has_feature(modules) && (__cplusplus <= 202302L)
// expected-error-re@-2{{declaration of 'rsize_t' must be imported from module '{{.+}}' before it is required}}
#else
// expected-error@-4{{unknown type}} expected-note@__stddef_size_t.h:*{{'size_t' declared here}}
#endif
wchar_t wc1;
void* v1 = NULL;
nullptr_t n1;
// unreachable() is declared in <utility> in C++
// https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2826.pdf suggests that it maybe should also be
// declared in <stddef.h> too, but it currently isn't.
static void f1(void) { unreachable(); } // expected-error 0+ {{undeclared identifier}}
max_align_t m1;
#if __cplusplus < 201103L
// expected-error@-2{{unknown type}}
#endif
size_t o1 = offsetof(struct astruct, member);
wint_t wi1; // expected-error{{unknown type}}

// rsize_t needs to be opted into via __STDC_WANT_LIB_EXT1__ >= 1.
#define __STDC_WANT_LIB_EXT1__ 1
#include <stddef.h>
ptrdiff_t p2;
size_t s2;
rsize_t r2;
wchar_t wc2;
void* v2 = NULL;
nullptr_t n2;
static void f2(void) { unreachable(); } // expected-error 0+ {{undeclared identifier}}
max_align_t m2;
#if __cplusplus < 201103L
// expected-error@-2{{unknown type}}
#endif
size_t o2 = offsetof(struct astruct, member);
wint_t wi2; // expected-error{{unknown type}}
