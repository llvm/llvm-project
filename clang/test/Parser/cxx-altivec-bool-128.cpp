// RUN: %clang_cc1 -triple=powerpc64-unknown-linux-gnu \
// RUN:            -target-feature +altivec -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu \
// RUN:            -target-feature +altivec -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu -target-cpu pwr10 \
// RUN:            -target-feature +altivec -target-feature -vsx \
// RUN:            -target-feature -power10-vector -fsyntax-only -verify %s

#include <altivec.h>

// Test 'vector bool __int128' type.

// These should have errors.
__vector bool __int128 v1_bi128;          // expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
__vector __bool __int128 v2_bi128;        // expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
vector bool __int128 v3_bi128;            // expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
vector __bool __int128 v4_bi128;          // expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
__vector bool unsigned __int128 v5_bi128; // expected-error {{cannot use 'unsigned' with '__vector bool'}} expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
__vector bool signed __int128 v6_bi128;   // expected-error {{cannot use 'signed' with '__vector bool'}} expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
vector bool unsigned __int128 v7_bi128;   // expected-error {{cannot use 'unsigned' with '__vector bool'}} expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
vector bool signed __int128 v8_bi128;     // expected-error {{cannot use 'signed' with '__vector bool'}} expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
__vector __bool signed __int128 v9_bi128; // expected-error {{cannot use 'signed' with '__vector bool'}} expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
vector __bool signed __int128 v10_bi128;  // expected-error {{cannot use 'signed' with '__vector bool'}} expected-error {{use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)}} expected-error {{use of '__int128' with '__vector bool' requires VSX support enabled (on POWER10 or later)}}
