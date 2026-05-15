// RUN: %clang_cc1 -triple x86_64-win32 -fdeclspec -verify %s
// RUN: %clang_cc1 -triple x86_64-mingw32 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -verify -fdeclspec %s
// RUN: %clang_cc1 -triple x86_64-win32-macho -verify -fdeclspec %s

extern __declspec(selectany) const int x1 = 1; // no warning, const means we need extern in C++

// Should we really warn on this?
extern __declspec(selectany) int x2 = 1; // expected-warning {{'extern' variable has an initializer}}

__declspec(selectany) void x3(void) { } // expected-error {{'selectany' attribute only applies to variable declarations with external linkage}}

void t() {
    __declspec(selectany) extern int i;
}
