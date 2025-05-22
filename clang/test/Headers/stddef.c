// RUN: rm -fR %t
// RUN: %clang_cc1 -fsyntax-only -verify=c99 -std=c99 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c11 -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c99-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c99 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c11-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c23-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c23 %s

struct astruct { char member; };

ptrdiff_t p0; // c99-error{{unknown type name 'ptrdiff_t'}} c11-error{{unknown type}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}
size_t s0; // c99-error{{unknown type name 'size_t'}} c11-error{{unknown type}} c23-error{{unknown type}} \
              c99-modules-error{{unknown type}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}
rsize_t r0; // c99-error{{unknown type name 'rsize_t'}} c11-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}
wchar_t wc0; // c99-error{{unknown type name 'wchar_t'}} c11-error{{unknown type}} c23-error{{unknown type}} \
                c99-modules-error{{unknown type}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}
void *v0 = NULL; // c99-error{{use of undeclared identifier 'NULL'}} c11-error{{undeclared identifier}} c23-error{{undeclared identifier}} \
                    c99-modules-error{{undeclared identifier}} c11-modules-error{{undeclared identifier}} c23-modules-error{{undeclared identifier}}
nullptr_t n0; // c99-error{{unknown type name 'nullptr_t'}} c11-error{{unknown type}} c23-error{{unknown type}} \
                 c99-modules-error{{unknown type}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}
static void f0(void) { unreachable(); } // c99-error{{call to undeclared function 'unreachable'}} c11-error{{undeclared function}} c23-error{{undeclared identifier}} \
                                           c99-modules-error{{undeclared function}} c11-modules-error{{undeclared function}} c23-modules-error{{undeclared identifier}}
max_align_t m0; // c99-error{{unknown type name 'max_align_t'}} c11-error{{unknown type}} c23-error{{unknown type}} \
                   c99-modules-error{{unknown type}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}
size_t o0 = offsetof(struct astruct, member); // c99-error{{unknown type name 'size_t'}} c99-error{{call to undeclared function 'offsetof'}} c99-error{{expected expression}} c99-error{{use of undeclared identifier 'member'}} \
                                                 c11-error{{unknown type}} c11-error{{undeclared function}} c11-error{{expected expression}} c11-error{{undeclared identifier}} \
                                                 c23-error{{unknown type}} c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}} \
                                                 c99-modules-error{{unknown type}} c99-modules-error{{undeclared function}} c99-modules-error{{expected expression}} c99-modules-error{{undeclared identifier}} \
                                                 c11-modules-error{{unknown type}} c11-modules-error{{undeclared function}} c11-modules-error{{expected expression}} c11-modules-error{{undeclared identifier}} \
                                                 c23-modules-error{{unknown type}} c23-modules-error{{undeclared identifier}} c23-modules-error{{expected expression}} c23-modules-error{{undeclared identifier}}
wint_t wi0; // c99-error{{unknown type name 'wint_t'}} c11-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type name 'wint_t'}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}

#include <stddef.h>

ptrdiff_t p1;
size_t s1;
rsize_t r1; // c99-error{{unknown type}} c11-error{{unknown type}} c23-error{{unknown type}} \
               c99-note@__stddef_size_t.h:*{{'size_t' declared here}} c11-note@__stddef_size_t.h:*{{'size_t' declared here}} c23-note@__stddef_size_t.h:*{{'size_t' declared here}} \
               c99-modules-error{{'rsize_t' must be declared before it is used}} c11-modules-error{{must be declared}} c23-modules-error{{must be declared}} \
               c99-modules-note@__stddef_rsize_t.h:*{{declaration here is not visible}} c11-modules-note@__stddef_rsize_t.h:*{{declaration here is not visible}} c23-modules-note@__stddef_rsize_t.h:*{{declaration here is not visible}}
wchar_t wc1;
void *v1 = NULL;
nullptr_t n1; // c99-error{{unknown type}} c11-error{{unknown type}} \
                 c99-modules-error{{unknown type}} c11-modules-error{{unknown type}}
static void f1(void) { unreachable(); } // c99-error{{undeclared function}} c11-error{{undeclared function}} \
                                           c99-modules-error{{undeclared function}} c11-modules-error{{undeclared function}}
max_align_t m1; // c99-error{{unknown type}} c99-modules-error{{'max_align_t' must be declared before it is used}} \
                   c99-modules-note@__stddef_max_align_t.h:*{{declaration here is not visible}}
size_t o1 = offsetof(struct astruct, member);
wint_t wi1; // c99-error{{unknown type}} c11-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}

// rsize_t needs to be opted into via __STDC_WANT_LIB_EXT1__ >= 1.
#define __STDC_WANT_LIB_EXT1__ 1
#include <stddef.h>
ptrdiff_t p2;
size_t s2;
rsize_t r2;
wchar_t wc2;
void *v2 = NULL;
nullptr_t n2; // c99-error{{unknown type}} c11-error{{unknown type}} \
                 c99-modules-error{{unknown type}} c11-modules-error{{unknown type}}
static void f2(void) { unreachable(); } // c99-error{{undeclared function}} c11-error{{undeclared function}} \
                                           c99-modules-error{{undeclared function}} c11-modules-error{{undeclared function}}
max_align_t m2; // c99-error{{unknown type}}
size_t o2 = offsetof(struct astruct, member);
wint_t wi2; // c99-error{{unknown type}} c11-error{{unknown type}} c23-error{{unknown type}} \
               c99-modules-error{{unknown type}} c11-modules-error{{unknown type}} c23-modules-error{{unknown type}}

// m2 and wi2 don't generate errors in modules, the "must be declared before used"
// errors are only emitted the first time the known-but-not-visible type is seen.
