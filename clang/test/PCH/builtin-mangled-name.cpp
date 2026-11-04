// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -include %s -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -include-pch %t -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -include-pch %t -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

constexpr const char *IntName = __builtin_mangled_name(int);

template <class T> constexpr const char *name() {
  return __builtin_mangled_name(T);
}

template <class T> constexpr bool is_int() {
  return __is_same(T, int);
}

#else

static_assert(__builtin_strcmp(IntName, "_ZTSi") == 0);
static_assert(__builtin_strcmp(name<long>(), "_ZTSl") == 0);
static_assert(is_int<int>());
static_assert(!is_int<long>());

#endif
