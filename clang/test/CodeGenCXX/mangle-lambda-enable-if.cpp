// RUN: %clang_cc1 -std=c++20 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

// Mangling a lambda in an enable_if condition must not recurse infinitely.
// https://github.com/llvm/llvm-project/issues/233807
template <int = 0>
int get() __attribute__((enable_if([]() { return true; }(), ""))) { return 0; }

// CHECK-LABEL: define {{.*}} @__cxx_global_var_init
// CHECK: call {{.*}} @_Z3getILi0EEUa9enable_ifIXclLUlvE_EEEiv()
int y = get();
