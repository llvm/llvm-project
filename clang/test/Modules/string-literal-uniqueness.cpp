// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cpp \
// RUN:  -o %t/A.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/b.cpp \
// RUN:  -fmodule-file=A=%t/A.pcm -o %t/B.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/c.cpp \
// RUN:  -fmodule-file=A=%t/A.pcm -o %t/C.pcm

// RUN: %clang_cc1 -std=c++20 -verify %t/main.cpp \
// RUN:  -fmodule-file=A=%t/A.pcm \
// RUN:  -fmodule-file=B=%t/B.pcm \
// RUN:  -fmodule-file=C=%t/C.pcm

// expected-no-diagnostics

//--- a.cpp

export module A;
export consteval const char *hello() { return "hello"; }
export constexpr const char *helloA0 = hello();
export constexpr const char *helloA1 = helloA0;
export constexpr const char *helloA2 = hello();

//--- b.cpp

export module B;
import A;
export constexpr const char *helloB1 = helloA0;
export constexpr const char *helloB2 = hello();

//--- c.cpp

export module C;
import A;
export constexpr const char *helloC1 = helloA1;
export constexpr const char *helloC2 = hello();

//--- main.cpp

import A;
import B;
import C;

// These are valid: they refer to the same evaluation of the same constant.
static_assert(helloA0 == helloA1);
static_assert(helloA0 == helloB1);
static_assert(helloA0 == helloC1);

// These refer to distinct evaluations, and so may or may not be equal.
static_assert(helloA1 == helloA2); // expected-error {{}} expected-note {{unspecified value}}
static_assert(helloA1 == helloB2); // expected-error {{}} expected-note {{unspecified value}}
static_assert(helloA1 == helloC2); // expected-error {{}} expected-note {{unspecified value}}
static_assert(helloA2 == helloB2); // expected-error {{}} expected-note {{unspecified value}}
static_assert(helloA2 == helloC2); // expected-error {{}} expected-note {{unspecified value}}
static_assert(helloB2 == helloC2); // expected-error {{}} expected-note {{unspecified value}}
