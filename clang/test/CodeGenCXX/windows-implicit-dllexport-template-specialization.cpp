// RUN: %clang_cc1 -std=c++11 -triple i686-windows         -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MS
// RUN: %clang_cc1 -std=c++11 -triple i686-windows-itanium -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-IA
// RUN: %clang_cc1 -std=c++11 -triple x86_64-scei-ps4      -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-PS4
// RUN: %clang_cc1 -std=c++11 -triple x86_64-sie-ps5       -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-PS4

template <typename>
struct s {};

template <typename T_>
class t : s<T_> {};

extern template class t<char>;
template class __declspec(dllexport) t<char>;

// CHECK-MS: dllexport {{.*}} @"??4?$t@D@@QAEAAV0@ABV0@@Z"
// CHECK-MS: dllexport {{.*}} @"??4?$s@D@@QAEAAU0@ABU0@@Z"

// CHECK-IA: dllexport {{.*}} @_ZN1tIcEaSERKS0_
// CHECK-IA: dllexport {{.*}} @_ZN1sIcEaSERKS0_

// CHECK-PS4-NOT: @_ZN1tIcEaSERKS0_
// CHECK-PS4-NOT: @_ZN1sIcEaSERKS0_
