// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-windows-msvc -fdeclspec -verify
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-windows-msvc -fdeclspec -verify
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-scei-ps4 -fdeclspec -verify

// MSVC emits this error too.
const int __declspec(selectany) test1 = 0; // expected-error {{'selectany' can only be applied to data items with external linkage}}

extern const int test2;
const int test2 = 42; // expected-note {{previous definition is here}}
extern __declspec(selectany) const int test2; // expected-warning {{attribute declaration must precede definition}}

extern const int test3;
const int __declspec(selectany) test3 = 42; // Standard usage.

struct Test4 {
  static constexpr int sdm = 0;
};
__declspec(selectany) constexpr int Test4::sdm; // no warning
