// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/a-part.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
//
// Test again with reduced BMI
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/a-part.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/a.cc -fmodule-file=a=%t/a.pcm -fsyntax-only -verify


//--- a.cppm
export module a;

constexpr int x = 43;

export constexpr int f() { return x; }

export template <typename T>
constexpr T g() {
    return x;
}

namespace nn {

constexpr int x = 88;

export constexpr int f() { return x; }

export template <typename T>
constexpr T g() {
    return x;
}
}

//--- use.cc
// expected-no-diagnostics
import a;

static_assert(f() == 43, "");

constexpr int x = 99;

static_assert(g<int>() == 43, "");

static_assert(x == 99, "");

namespace nn {
static_assert(f() == 88, "");

constexpr int x = 1000;

static_assert(g<int>() == 88, "");

static_assert(x == 1000, "");

}

//--- a-part.cppm
module a:impl;
import a;

static_assert(x == 43, "");

constexpr int x = 1000; // expected-error {{redefinition of 'x'}}
                        // expected-note@* {{previous definition is here}}

//--- a.cc
module a;

static_assert(x == 43, "");

constexpr int x = 1000; // expected-error {{redefinition of 'x'}}
                        // expected-note@* {{previous definition is here}}

