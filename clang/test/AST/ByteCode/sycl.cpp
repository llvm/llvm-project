// RUN: %clang_cc1 %s -std=c++17 -triple x86_64-linux-gnu -fsycl-is-device -verify=both,ref -fsyntax-only -Wno-unused
// RUN: %clang_cc1 %s -std=c++17 -triple x86_64-linux-gnu -fsycl-is-device -verify=both,expected -fsyntax-only -Wno-unused -fexperimental-new-constant-interpreter

// both-no-diagnostics

constexpr int a = 0;
constexpr const char *a_name = __builtin_sycl_unique_stable_name(decltype(a));
static_assert(__builtin_strcmp(a_name, "_ZTSKi") == 0);

