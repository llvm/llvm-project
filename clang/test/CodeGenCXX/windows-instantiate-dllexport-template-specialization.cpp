// RUN: %clang_cc1 -triple i686-windows         -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MS
// RUN: %clang_cc1 -triple i686-windows-itanium -fdeclspec -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-scei-ps4      -fdeclspec -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-sie-ps5       -fdeclspec -emit-llvm %s -o - | FileCheck %s

struct s {
  template <bool b = true> static bool f();
};

template <typename T> bool template_using_f(T) { return s::f(); }

bool use_template_using_f() { return template_using_f(0); }

template<>
bool __declspec(dllexport) s::f<true>() { return true; }

// CHECK-MS: dllexport {{.*}} @"??$f@$00@s@@SA_NXZ"
// CHECK: dllexport {{.*}} @_ZN1s1fILb1EEEbv
