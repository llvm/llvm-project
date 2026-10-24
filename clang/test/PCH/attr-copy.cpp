// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -include %s -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -fsyntax-only -verify %s

#ifndef HEADER
#define HEADER

void source(int *) __attribute__((nonnull(1)));
void copied(int *) __attribute__((copy(source)));
template <void (*F)(int *)> struct FunctionCopy {
  static void function(int *) __attribute__((copy(F)));
};
using Callback = int (__attribute__((ms_abi)) *)(int);
template <typename T> Callback make_callback() {
  return [](int x) __attribute__((copy((Callback)nullptr))) { return x; };
}

#else

void copied_from_pch(int *) __attribute__((copy(source)));
Callback lambda_from_pch =
    [](int x) __attribute__((copy((Callback)nullptr))) { return x; };
Callback instantiated_from_pch = make_callback<int>();

void use() {
  copied(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
  copied_from_pch(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
  FunctionCopy<source>::function(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
}

#endif
