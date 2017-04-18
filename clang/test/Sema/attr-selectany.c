// RUN: %clang_cc1 -triple x86_64-win32 -fdeclspec -verify %s
// RUN: %clang_cc1 -triple x86_64-mingw32 -verify %s

extern __declspec(selectany) const int x1 = 1; // no warning, const means we need extern in C++

// Should we really warn on this?
extern __declspec(selectany) int x2 = 1; // expected-warning {{'extern' variable has an initializer}}

__declspec(selectany) void foo() { } // expected-error{{'selectany' can only be applied to data items with external linkage}}
