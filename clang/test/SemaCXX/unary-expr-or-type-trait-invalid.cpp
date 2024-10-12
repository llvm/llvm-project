// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value %s
//
// Note: This test is ensure the code does not cause a crash as previously
// reported in (#GH111594). The specific diagnostics are unimportant.

a() {struct b c (sizeof(b * [({ {tree->d* next)} 0

// expected-error@6 0+{{}}
// expected-error@11 0+{{}}
// expected-note@6 0+{{}}

