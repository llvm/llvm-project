// RUN: %clang_cc1 -E %s -triple x86_64-linux-gnu -x c++ -o - | FileCheck %s --check-prefix=NO-RELATIVE-VTABLE
// RUN: %clang_cc1 -E %s -triple x86_64-linux-gnu -x c++ -fexperimental-relative-c++-abi-vtables -o - | FileCheck %s --check-prefix=RELATIVE-VTABLE
// RUN: %clang_cc1 -E %s -triple x86_64-linux-gnu -x c++ -fno-experimental-relative-c++-abi-vtables -o - | FileCheck %s --check-prefix=NO-RELATIVE-VTABLE
// RUN: %clang_cc1 -E %s -triple x86_64-linux-gnu -x c -fexperimental-relative-c++-abi-vtables -o - | FileCheck %s --check-prefix=NO-RELATIVE-VTABLE

#if __has_feature(cxx_abi_relative_vtable)
int has_relative_vtable();
#else
int has_no_relative_vtable();
#endif

// RELATIVE-VTABLE: has_relative_vtable
// NO-RELATIVE-VTABLE: has_no_relative_vtable
