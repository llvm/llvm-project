// RUN: %clang_cc1 -fsyntax-only -verify=c99 -std=c99 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c11 -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 %s

struct astruct { char member; };

ptrdiff_t p0; // c99-error{{unknown type name 'ptrdiff_t'}} c11-error{{unknown type}} c23-error{{unknown type}}
size_t s0; // c99-error{{unknown type name 'size_t'}} c11-error{{unknown type}} c23-error{{unknown type}}
rsize_t r0; // c99-error{{unknown type name 'rsize_t'}} c11-error{{unknown type}} c23-error{{unknown type}}
wchar_t wc0; // c99-error{{unknown type name 'wchar_t'}} c11-error{{unknown type}} c23-error{{unknown type}}
void *v0 = NULL; // c99-error{{use of undeclared identifier 'NULL'}} c11-error{{undeclared identifier}} c23-error{{undeclared identifier}}
nullptr_t n0; // c99-error{{unknown type name 'nullptr_t'}} c11-error{{unknown type}} c23-error{{unknown type}}
static void f0(void) { unreachable(); } // c99-error{{call to undeclared function 'unreachable'}} c11-error{{undeclared function}} c23-error{{undeclared identifier}}
max_align_t m0; // c99-error{{unknown type name 'max_align_t'}} c11-error{{unknown type}} c23-error{{unknown type}}
size_t o0 = offsetof(struct astruct, member); // c99-error{{unknown type name 'size_t'}} c99-error{{call to undeclared function 'offsetof'}} c99-error{{expected expression}} c99-error{{use of undeclared identifier 'member'}} \
                                                 c11-error{{unknown type}} c11-error{{undeclared function}} c11-error{{expected expression}} c11-error{{undeclared identifier}} \
                                                 c23-error{{unknown type}} c23-error{{undeclared identifier}} c23-error{{expected expression}} c23-error{{undeclared identifier}}
wint_t wi0; // c99-error{{unknown type name 'wint_t'}} c11-error{{unknown type}} c23-error{{unknown type}}

#include <stddef.h>

ptrdiff_t p1;
size_t s1;
rsize_t r1; // c99-error{{unknown type}} c11-error{{unknown type}} c23-error{{unknown type}}
            // c99-note@__stddef_size_t.h:*{{'size_t' declared here}} c11-note@__stddef_size_t.h:*{{'size_t' declared here}} c23-note@__stddef_size_t.h:*{{'size_t' declared here}}
wchar_t wc1;
void *v1 = NULL;
nullptr_t n1; // c99-error{{unknown type}} c11-error{{unknown type}}
static void f1(void) { unreachable(); } // c99-error{{undeclared function}} c11-error{{undeclared function}}
max_align_t m1; // c99-error{{unknown type}}
size_t o1 = offsetof(struct astruct, member);
wint_t wi1; // c99-error{{unknown type}} c11-error{{unknown type}} c23-error{{unknown type}}

// rsize_t needs to be opted into via __STDC_WANT_LIB_EXT1__ >= 1.
#define __STDC_WANT_LIB_EXT1__ 1
#include <stddef.h>
ptrdiff_t p2;
size_t s2;
rsize_t r2;
wchar_t wc2;
void *v2 = NULL;
nullptr_t n2; // c99-error{{unknown type}} c11-error{{unknown type}}
static void f2(void) { unreachable(); } // c99-error{{undeclared function}} c11-error{{undeclared function}}
max_align_t m2; // c99-error{{unknown type}}
size_t o2 = offsetof(struct astruct, member);
wint_t wi2; // c99-error{{unknown type}} c11-error{{unknown type}} c23-error{{unknown type}}
