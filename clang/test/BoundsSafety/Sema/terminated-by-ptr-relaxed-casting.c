
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=strict,both %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=relaxed,both -Wno-error=bounds-safety-strict-terminated-by-cast -Wno-error %s

#include <ptrcheck.h>

// The relaxed case should be removed by rdar://118390724

void foo(const char * __null_terminated); // both-note{{passing argument to parameter here}}
void bar(const char * __null_terminated * __single); // both-note{{passing argument to parameter here}}

void test(const char * __single sp) {
    // strict-error@+2{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()}}
    // relaxed-warning@+1{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()}}
    const char * __null_terminated ntp = sp;
    // strict-error@+2{{casting 'const char *__single' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // relaxed-warning@+1{{casting 'const char *__single' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    const char * __null_terminated ntp2 = (const char * __null_terminated) sp;

    // strict-error@+2{{passing 'const char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // relaxed-warning@+1{{passing 'const char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    foo(sp);
    // strict-error@+2{{casting 'const char *__single' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // relaxed-warning@+1{{casting 'const char *__single' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    foo((const char * __null_terminated) sp);

    const char * __null_terminated ntp3 = ntp;
    // strict-error@+2{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // relaxed-warning@+1{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    ntp3 = sp;
    // strict-error@+2{{casting 'const char *__single' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // relaxed-warning@+1{{casting 'const char *__single' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    ntp3 = (const char * __null_terminated) sp;


    /* --- Nested --- */

    const char * __single * __single spp = &sp;
    // strict-error@+2{{initializing 'const char *__single __terminated_by(0)*__single' (aka 'const char *__single*__single') with an expression of incompatible type 'const char *__single*__single' that adds '__terminated_by' attribute is not allowed}}
    // relaxed-warning@+1{{initializing 'const char *__single __terminated_by(0)*__single' (aka 'const char *__single*__single') with an expression of incompatible type 'const char *__single*__single' that adds '__terminated_by' attribute is not allowed}}
    const char * __null_terminated * __single ntpp = spp;
    // strict-error@+2{{casting 'const char *__single*__single' to incompatible type 'const char * __terminated_by(0)*__single' (aka 'const char **__single') that adds '__terminated_by' attribute is not allowed}}
    // relaxed-warning@+1{{casting 'const char *__single*__single' to incompatible type 'const char * __terminated_by(0)*__single' (aka 'const char **__single') that adds '__terminated_by' attribute is not allowed}}
    const char * __null_terminated * __single ntpp2 = (const char * __null_terminated * __single) spp;

    // strict-error@+2{{passing 'const char *__single*__single' to parameter of incompatible type 'const char *__single __terminated_by(0)*__single' (aka 'const char *__single*__single') that adds '__terminated_by' attribute is not allowed}}
    // relaxed-warning@+1{{passing 'const char *__single*__single' to parameter of incompatible type 'const char *__single __terminated_by(0)*__single' (aka 'const char *__single*__single') that adds '__terminated_by' attribute is not allowed}}
    bar(spp);
    // strict-error@+2{{casting 'const char *__single*__single' to incompatible type 'const char * __terminated_by(0)*__single' (aka 'const char **__single') that adds '__terminated_by' attribute is not allowed}}
    // relaxed-warning@+1{{casting 'const char *__single*__single' to incompatible type 'const char * __terminated_by(0)*__single' (aka 'const char **__single') that adds '__terminated_by' attribute is not allowed}}
    bar((const char * __null_terminated * __single) spp);

    const char * __null_terminated * __single ntpp3 = ntpp;
    // strict-error@+2{{assigning to 'const char *__single __terminated_by(0)*__single' (aka 'const char *__single*__single') from incompatible type 'const char *__single*__single' that adds '__terminated_by' attribute is not allowed}}
    // relaxed-warning@+1{{assigning to 'const char *__single __terminated_by(0)*__single' (aka 'const char *__single*__single') from incompatible type 'const char *__single*__single' that adds '__terminated_by' attribute is not allowed}}
    ntpp3 = spp;
    // strict-error@+2{{casting 'const char *__single*__single' to incompatible type 'const char * __terminated_by(0)*__single' (aka 'const char **__single') that adds '__terminated_by' attribute is not allowed}}
    // relaxed-warning@+1{{casting 'const char *__single*__single' to incompatible type 'const char * __terminated_by(0)*__single' (aka 'const char **__single') that adds '__terminated_by' attribute is not allowed}}
    ntpp3 = (const char * __null_terminated * __single) spp;
}
