// RUN: rm -fR %t
// RUN: %clang_cc1 -fsyntax-only -verify=c99 -std=c99 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c99-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c99 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c23-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c23 %s

// Use C99 to verify that __need_ can be used to get types that wouldn't normally be available.

struct astruct { char member; };

ptrdiff_t p0; // c99-error{{unknown type name 'ptrdiff_t'}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}
size_t s0; // c99-error{{unknown type name 'size_t'}} c23-error{{unknown type}} \
              c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}
rsize_t r0; // c99-error{{unknown type name 'rsize_t'}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}
wchar_t wc0; // c99-error{{unknown type name 'wchar_t'}} c23-error{{unknown type}} \
                c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}
void *v0 = NULL; // c99-error{{use of undeclared identifier 'NULL'}} c23-error{{undeclared identifier}} \
                    c99-modules-error{{undeclared identifier}} c23-modules-error{{undeclared identifier}}
nullptr_t n0; // c99-error{{unknown type name 'nullptr_t'}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}
static void f0(void) { unreachable(); } // c99-error{{call to undeclared function 'unreachable'}} c23-error{{undeclared identifier 'unreachable'}} \
                                           c99-modules-error{{undeclared function}} c23-modules-error{{undeclared identifier}}
max_align_t m0; // c99-error{{unknown type name 'max_align_t'}} c23-error{{unknown type}} \
                   c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}
size_t o0 = offsetof(struct astruct, member); // c99-error{{unknown type name 'size_t'}} c99-error{{call to undeclared function 'offsetof'}} c99-error{{expected expression}} c99-error{{use of undeclared identifier 'member'}} \
                                                 c23-error{{unknown type name 'size_t'}} c23-error{{undeclared identifier 'offsetof'}} c23-error{{expected expression}} c23-error{{use of undeclared identifier 'member'}} \
                                                 c99-modules-error{{unknown type}} c99-modules-error{{undeclared function}} c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{unknown type}} c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi0; // c99-error{{unknown type name 'wint_t'}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#define __need_ptrdiff_t
#include <stddef.h>

ptrdiff_t p1;
size_t s1; // c99-error{{unknown type}} c23-error{{unknown type}} \
              c99-modules-error{{'size_t' must be declared before it is used}} c23-modules-error{{must be declared}} \
              c99-modules-note@__stddef_size_t.h:*{{declaration here is not visible}} c23-modules-note@__stddef_size_t.h:*{{declaration here is not visible}}
rsize_t r1; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{'rsize_t' must be declared before it is used}} c23-modules-error{{must be declared}} \
               c99-modules-note@__stddef_rsize_t.h:*{{declaration here is not visible}} c23-modules-note@__stddef_rsize_t.h:*{{declaration here is not visible}}
wchar_t wc1; // c99-error{{unknown type}} c23-error{{unknown type}} \
                c99-modules-error{{'wchar_t' must be declared before it is used}} c23-modules-error{{must be declared}} \
                c99-modules-note@__stddef_wchar_t.h:*{{declaration here is not visible}} c23-modules-note@__stddef_wchar_t.h:*{{declaration here is not visible}}
void *v1 = NULL; // c99-error{{undeclared identifier}} c23-error{{undeclared identifier}} \
                    c99-modules-error{{undeclared identifier}} c23-modules-error{{undeclared identifier}}
nullptr_t n1; // c99-error{{unknown type}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}} c23-modules-error{{'nullptr_t' must be declared before it is used}} \
                 c23-modules-note@__stddef_nullptr_t.h:*{{declaration here is not visible}}
static void f1(void) { unreachable(); } // c99-error{{undeclared function}} c23-error{{undeclared identifier}} \
                                           c99-modules-error{{undeclared function}} c23-modules-error{{undeclared identifier}}
max_align_t m1; // c99-error{{unknown type}} c23-error{{unknown type}} \
                   c99-modules-error{{'max_align_t' must be declared before it is used}} c23-modules-error{{must be declared}} \
                   c99-modules-note@__stddef_max_align_t.h:*{{declaration here is not visible}} c23-modules-note@__stddef_max_align_t.h:*{{declaration here is not visible}}
size_t o1 = offsetof(struct astruct, member); // c99-error{{unknown type}} c99-error{{expected expression}} c99-error{{undeclared identifier}} \
                                                 c23-error{{unknown type}} c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi1; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

// The "must be declared before used" errors are only emitted the first time a
// known-but-not-visible type is seen. At this point the _Builtin_stddef module
// has been built and all of the types tried, so most of the errors won't be
// repeated below in modules. The types still aren't available, just the errors
// aren't repeated. e.g. rsize_t still isn't available, if r1 above got deleted,
// its error would move to r2 below.

#define __need_size_t
#include <stddef.h>

ptrdiff_t p2;
size_t s2;
rsize_t r2; // c99-error{{unknown type}} c23-error{{unknown type}}
            // c99-note@__stddef_size_t.h:*{{'size_t' declared here}} c23-note@__stddef_size_t.h:*{{'size_t' declared here}}
wchar_t wc2; // c99-error{{unknown type}} c23-error{{unknown type}}
void *v2 = NULL; // c99-error{{undeclared identifier}} c23-error{{undeclared identifier}} \
                    c99-modules-error{{undeclared identifier}} c23-modules-error{{undeclared identifier}}
nullptr_t n2; // c99-error{{unknown type}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}}
static void f2(void) { unreachable(); } // c99-error{{undeclared function}} c23-error{{undeclared identifier}} \
                                           c99-modules-error{{undeclared function}} c23-modules-error{{undeclared identifier}}
max_align_t m2; // c99-error{{unknown type}} c23-error{{unknown type}}
size_t o2 = offsetof(struct astruct, member); // c99-error{{expected expression}} c99-error{{undeclared identifier}} \
                                                 c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi2; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#define __need_rsize_t
#include <stddef.h>

ptrdiff_t p3;
size_t s3;
rsize_t r3;
wchar_t wc3; // c99-error{{unknown type}} c23-error{{unknown type}}
void *v3 = NULL; // c99-error{{undeclared identifier}} c23-error{{undeclared identifier}} \
                    c99-modules-error{{undeclared identifier}} c23-modules-error{{undeclared identifier}}
nullptr_t n3; // c99-error{{unknown type}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}}
static void f3(void) { unreachable(); } // c99-error{{undeclared function}} c23-error{{undeclared identifier}} \
                                           c99-modules-error{{undeclared function}} c23-modules-error{{undeclared identifier}}
max_align_t m3; // c99-error{{unknown type}} c23-error{{unknown type}}
size_t o3 = offsetof(struct astruct, member); // c99-error{{expected expression}} c99-error{{undeclared identifier}} \
                                                 c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi3; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#define __need_wchar_t
#include <stddef.h>

ptrdiff_t p4;
size_t s4;
rsize_t r4;
wchar_t wc4;
void *v4 = NULL; // c99-error{{undeclared identifier}} c23-error{{undeclared identifier}} \
                    c99-modules-error{{undeclared identifier}} c23-modules-error{{undeclared identifier}}
nullptr_t n4; // c99-error{{unknown type}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}}
static void f4(void) { unreachable(); } // c99-error{{undeclared function}} c23-error{{undeclared identifier}} \
                                           c99-modules-error{{undeclared function}} c23-modules-error{{undeclared identifier}}
max_align_t m4; // c99-error{{unknown type}} c23-error{{unknown type}}
size_t o4 = offsetof(struct astruct, member); // c99-error{{expected expression}} c99-error{{undeclared identifier}} \
                                                 c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi4; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#define __need_NULL
#include <stddef.h>

ptrdiff_t p5;
size_t s5;
rsize_t r5;
wchar_t wc5;
void *v5 = NULL;
nullptr_t n5; // c99-error{{unknown type}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}}
static void f5(void) { unreachable(); } // c99-error{{undeclared function}} c23-error{{undeclared identifier}} \
                                           c99-modules-error{{undeclared function}} c23-modules-error{{undeclared identifier}}
max_align_t m5; // c99-error{{unknown type}} c23-error{{unknown type}}
size_t o5 = offsetof(struct astruct, member); // c99-error{{expected expression}} c99-error{{undeclared identifier}} \
                                                 c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi5; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

// nullptr_t doesn't get declared before C23 because its definition
// depends on nullptr.
#define __need_nullptr_t
#include <stddef.h>

ptrdiff_t p6;
size_t s6;
rsize_t r6;
wchar_t wc6;
void *v6 = NULL;
nullptr_t n6; // c99-error{{unknown type}} c99-modules-error{{unknown type}}
static void f6(void) { unreachable(); } // c99-error{{undeclared function}} c23-error{{undeclared identifier}} \
                                           c99-modules-error{{undeclared function}} c23-modules-error{{undeclared identifier}}
max_align_t m6; // c99-error{{unknown type}} c23-error{{unknown type}}
size_t o6 = offsetof(struct astruct, member); // c99-error{{expected expression}} c99-error{{undeclared identifier}} \
                                                 c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi6; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#define __need_unreachable
#include <stddef.h>

ptrdiff_t p7;
size_t s7;
rsize_t r7;
wchar_t wc7;
void *v7 = NULL;
nullptr_t n7 ; // c99-error{{unknown type}} c99-modules-error{{unknown type}}
static void f7(void) { unreachable(); }
max_align_t m7; // c99-error{{unknown type}} c23-error{{unknown type}}
size_t o7 = offsetof(struct astruct, member); // c99-error{{expected expression}} c99-error{{undeclared identifier}} \
                                                 c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi7; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#define __need_max_align_t
#include <stddef.h>

ptrdiff_t p8;
size_t s8;
rsize_t r8;
wchar_t wc8;
void *v8 = NULL;
nullptr_t n8; // c99-error{{unknown type}} c99-modules-error{{unknown type}}
static void f8(void) { unreachable(); }
max_align_t m8;
size_t o8 = offsetof(struct astruct, member); // c99-error{{expected expression}} c99-error{{undeclared identifier}} \
                                                 c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi8; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#define __need_offsetof
#include <stddef.h>

ptrdiff_t p9;
size_t s9;
rsize_t r9;
nullptr_t n9; // c99-error{{unknown type}} c99-modules-error{{unknown type}}
static void f9(void) { unreachable(); }
wchar_t wc9;
void *v9 = NULL;
max_align_t m9;
size_t o9 = offsetof(struct astruct, member);
wint_t wi9; // c99-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#define __need_wint_t
#include <stddef.h>

ptrdiff_t p10;
size_t s10;
rsize_t r10;
wchar_t wc10;
void *v10 = NULL;
nullptr_t n10; // c99-error{{unknown type}} c99-modules-error{{unknown type}}
static void f10(void) { unreachable(); }
max_align_t m10;
size_t o10 = offsetof(struct astruct, member);
wint_t wi10;
