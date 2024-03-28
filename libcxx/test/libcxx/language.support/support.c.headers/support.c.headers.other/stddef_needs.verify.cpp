//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is the same test as clang/test/Headers/stddefneeds.c, but to test the
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

#define __need_ptrdiff_t
#include <stddef.h>

ptrdiff_t p1;
size_t s1;
#if __has_feature(modules) && (__cplusplus <= 202302L)
// expected-error-re@-2{{declaration of 'size_t' must be imported from module '{{.+}}' before it is required}}
#else
// expected-error@-4{{unknown type}}
#endif
rsize_t r1;
#if __has_feature(modules) && (__cplusplus <= 202302L)
// expected-error-re@-2{{declaration of 'rsize_t' must be imported from module '{{.+}}' before it is required}}
#else
// expected-error@-4{{unknown type}}
#endif
wchar_t wc1;
void* v1 = NULL;                        // expected-error{{undeclared identifier}}
nullptr_t n1;                           // expected-error{{unknown type}}
static void f1(void) { unreachable(); } // expected-error{{undeclared identifier}}
max_align_t m1;
#if __has_feature(modules) && (__cplusplus <= 202302L)
// expected-error-re@-2{{declaration of 'max_align_t' must be imported from module '{{.+}}' before it is required}}
#else
// expected-error@-4{{unknown type}}
#endif
size_t o1 = offsetof(struct astruct, member);
#if !__has_feature(modules) || (__cplusplus > 202302L)
// expected-error@-2{{unknown type}}
#endif
// expected-error@-4{{expected expression}} expected-error@-4{{undeclared identifier}}
wint_t wi1; // expected-error{{unknown type}}

// The "declaration must be imported" errors are only emitted the first time a
// known-but-not-visible type is seen. At this point the _Builtin_stddef module
// has been built and all of the types tried, so most of the errors won't be
// repeated below in modules. The types still aren't available, just the errors
// aren't repeated. e.g. rsize_t still isn't available, if r1 above got deleted,
// its error would move to r2 below.

#define __need_size_t
#include <stddef.h>

ptrdiff_t p2;
size_t s2;
rsize_t r2;
#if !__has_feature(modules) || (__cplusplus > 202302L)
// expected-error@-2{{unknown type}}
#endif
wchar_t wc2;
void* v2 = NULL;                        // expected-error{{undeclared identifier}}
nullptr_t n2;                           // expected-error{{unknown type}}
static void f2(void) { unreachable(); } // expected-error{{undeclared identifier}}
max_align_t m2;
#if !__has_feature(modules) || (__cplusplus > 202302L)
// expected-error@-2{{unknown type}}
#endif
size_t o2 =
    offsetof(struct astruct, member); // expected-error{{expected expression}} expected-error{{undeclared identifier}}
wint_t wi2;                           // expected-error{{unknown type}}

#define __need_rsize_t
#include <stddef.h>

ptrdiff_t p3;
size_t s3;
rsize_t r3;
wchar_t wc3;
void* v3 = NULL;                        // expected-error{{undeclared identifier}}
nullptr_t n3;                           // expected-error{{unknown type}}
static void f3(void) { unreachable(); } // expected-error{{undeclared identifier}}
max_align_t m3;
#if !__has_feature(modules) || (__cplusplus > 202302L)
// expected-error@-2{{unknown type}}
#endif
size_t o3 =
    offsetof(struct astruct, member); // expected-error{{expected expression}} expected-error{{undeclared identifier}}
wint_t wi3;                           // expected-error{{unknown type}}

#define __need_wchar_t
#include <stddef.h>

ptrdiff_t p4;
size_t s4;
rsize_t r4;
wchar_t wc4;
void* v4 = NULL;                        // expected-error{{undeclared identifier}}
nullptr_t n4;                           // expected-error{{unknown type}}
static void f4(void) { unreachable(); } // expected-error{{undeclared identifier}}
max_align_t m4;
#if !__has_feature(modules) || (__cplusplus > 202302L)
// expected-error@-2{{unknown type}}
#endif
size_t o4 =
    offsetof(struct astruct, member); // expected-error{{expected expression}} expected-error{{undeclared identifier}}
wint_t wi4;                           // expected-error{{unknown type}}

#define __need_NULL
#include <stddef.h>

ptrdiff_t p5;
size_t s5;
rsize_t r5;
wchar_t wc5;
void* v5 = NULL;
nullptr_t n5;                           // expected-error{{unknown type}}
static void f5(void) { unreachable(); } // expected-error{{undeclared identifier}}
max_align_t m5;
#if !__has_feature(modules) || (__cplusplus > 202302L)
// expected-error@-2{{unknown type}}
#endif
size_t o5 =
    offsetof(struct astruct, member); // expected-error{{expected expression}} expected-error{{undeclared identifier}}
wint_t wi5;                           // expected-error{{unknown type}}

// nullptr_t doesn't get declared before C23 because its definition
// depends on nullptr.
#define __need_nullptr_t
#include <stddef.h>

ptrdiff_t p6;
size_t s6;
rsize_t r6;
wchar_t wc6;
void* v6 = NULL;
nullptr_t n6;
static void f6(void) { unreachable(); } // expected-error{{undeclared identifier}}
max_align_t m6;
#if !__has_feature(modules) || (__cplusplus > 202302L)
// expected-error@-2{{unknown type}}
#endif
size_t o6 =
    offsetof(struct astruct, member); // expected-error{{expected expression}} expected-error{{undeclared identifier}}
wint_t wi6;                           // expected-error{{unknown type}}

#define __need_unreachable
#include <stddef.h>

ptrdiff_t p7;
size_t s7;
rsize_t r7;
wchar_t wc7;
void* v7 = NULL;
nullptr_t n7;
// __need_unreachable currently declares unreachable(), but the C++23 standard only lists unreachable() in <utility>
// so maybe stddef.h shouldn't declare it even with __need_unreachable.
static void f7(void) { unreachable(); } // expected-error 0+ {{undeclared identifier}}
max_align_t m7;
#if !__has_feature(modules) || (__cplusplus > 202302L)
// expected-error@-2{{unknown type}}
#endif
size_t o7 =
    offsetof(struct astruct, member); // expected-error{{expected expression}} expected-error{{undeclared identifier}}
wint_t wi7;                           // expected-error{{unknown type}}

#define __need_max_align_t
#include <stddef.h>

ptrdiff_t p8;
size_t s8;
rsize_t r8;
wchar_t wc8;
void* v8 = NULL;
nullptr_t n8;
static void f8(void) { unreachable(); } // expected-error 0+ {{undeclared identifier}}
max_align_t m8;
size_t o8 =
    offsetof(struct astruct, member); // expected-error{{expected expression}} expected-error{{undeclared identifier}}
wint_t wi8;                           // expected-error{{unknown type}}

#define __need_offsetof
#include <stddef.h>

ptrdiff_t p9;
size_t s9;
rsize_t r9;
nullptr_t n9;
static void f9(void) { unreachable(); } // expected-error 0+ {{undeclared identifier}}
wchar_t wc9;
void* v9 = NULL;
max_align_t m9;
size_t o9 = offsetof(struct astruct, member);
wint_t wi9; // expected-error{{unknown type}}

#define __need_wint_t
#include <stddef.h>

ptrdiff_t p10;
size_t s10;
rsize_t r10;
wchar_t wc10;
void* v10 = NULL;
nullptr_t n10;
static void f10(void) { unreachable(); } // expected-error 0+ {{undeclared identifier}}
max_align_t m10;
size_t o10 = offsetof(struct astruct, member);
wint_t wi10;
