// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 O.cpp \
// RUN:    -emit-module-interface -o O.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 O.pcm -S -emit-llvm \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK-O

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 P.cpp \
// RUN:    -emit-module-interface -fmodule-file=O.pcm -o P.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 P.pcm -S -emit-llvm \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK-P

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 Q.cpp \
// RUN:    -emit-module-interface -fmodule-file=O.pcm -o Q.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 Q.pcm -S -emit-llvm \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK-Q

// Testing cases where we can elide the module initializer guard variable.

// This module has no global inits and does not import any other module
//--- O.cpp

export module O;

export int foo ();

// CHECK-O: define void @_ZGIW1O
// CHECK-O-LABEL: entry
// CHECK-O-NEXT: ret void
// CHECK-O-NOT: @_ZGIW1O__in_chrg

// This has no global inits but imports a module, and therefore needs a guard
// variable.
//--- P.cpp

export module P;

export import O;
export int bar ();

// CHECK-P: define void @_ZGIW1P
// CHECK-P-LABEL: init
// CHECK-P: store i8 1, ptr @_ZGIW1P__in_chrg
// CHECK-P: call void @_ZGIW1O()
// CHECK-P-NOT: call void @__cxx_global_var_init

// This imports a module and has global inits, so needs a guard.
//--- Q.cpp

export module Q;
export import O;

export struct Quack {
  Quack(){};
};

export Quack Duck;

export int baz ();

// CHECK-Q: define internal void @__cxx_global_var_init
// CHECK-Q: call {{.*}} @_ZNW1Q5QuackC1Ev
// CHECK-Q: define void @_ZGIW1Q
// CHECK-Q: store i8 1, ptr @_ZGIW1Q__in_chrg
// CHECK-Q: call void @_ZGIW1O()
// CHECK-Q: call void @__cxx_global_var_init

