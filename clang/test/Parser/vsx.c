// RUN: %clang_cc1 -triple=powerpc64-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -fsyntax-only -verify %s

// Legitimate for VSX.
__vector double vv_d1;
vector double v_d2;

// These should have errors.
__vector long double  vv_ld3;        // expected-error {{cannot use 'long double' with '__vector'}}
vector long double  v_ld4;           // expected-error {{cannot use 'long double' with '__vector'}}
