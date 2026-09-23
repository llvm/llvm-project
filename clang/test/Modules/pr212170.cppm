// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-module-interface %t/test.cppm -o %t/test.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-llvm %t/test.pcm -o - | FileCheck %t/test.cppm \
// RUN:   --implicit-check-not=inline_bar \
// RUN:   --implicit-check-not=static_inline_bar
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fmodule-file=test=%t/test.pcm -emit-llvm %t/use.cpp -o - | \
// RUN:   FileCheck %t/use.cpp

// https://github.com/llvm/llvm-project/issues/212170

//--- test.cppm

export module test;

int side_effect();

static int bar = side_effect();
[[gnu::used]] static int used_bar = side_effect();

inline int inline_bar = side_effect();
static inline int static_inline_bar = side_effect();

// CHECK-DAG: @_ZL3bar = internal global i32 0
// CHECK-DAG: @_ZL8used_bar = internal global i32 0

// CHECK-LABEL: define internal void @__cxx_global_var_init()
// CHECK: call{{.*}} @_ZW4test11side_effectv()
// CHECK: store i32 {{.*}}, ptr @_ZL3bar

// CHECK-LABEL: define internal void @__cxx_global_var_init.1()
// CHECK: call{{.*}} @_ZW4test11side_effectv()
// CHECK: store i32 {{.*}}, ptr @_ZL8used_bar

// CHECK-LABEL: define void @_ZGIW4test()
// CHECK: call void @__cxx_global_var_init()
// CHECK: call void @__cxx_global_var_init.1()

//--- use.cpp

import test;

int main() {}

// The consumer calls the module initializer but does not emit copies of the
// module's internal variables or their initialization functions.
// CHECK-NOT: @_ZL3bar
// CHECK-NOT: @_ZL8used_bar
// CHECK: declare void @_ZGIW4test()
// CHECK-LABEL: define internal void @_GLOBAL__sub_I_use.cpp()
// CHECK: call void @_ZGIW4test()
