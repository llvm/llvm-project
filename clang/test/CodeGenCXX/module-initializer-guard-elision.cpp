// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 O.cpp \
// RUN:    -emit-module-interface -o O.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 O.pcm -S -emit-llvm \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK-O

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 P.cpp \
// RUN:    -emit-module-interface -fmodule-file=O=O.pcm -o P.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 P.pcm -S -emit-llvm \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK-P

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 Q.cpp \
// RUN:    -emit-module-interface -o Q.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 Q.pcm -S -emit-llvm \
// RUN:    -o - | FileCheck %s --check-prefix=CHECK-Q

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 R.cpp \
// RUN:    -emit-module-interface -fmodule-file=Q=Q.pcm -o R.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 R.pcm -S -emit-llvm \
// RUN:    -o - | FileCheck %s --check-prefix=CHECK-R

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 S.cpp \
// RUN:    -emit-module-interface -fmodule-file=Q=Q.pcm -fmodule-file=R=R.pcm -o S.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 S.pcm -S -emit-llvm \
// RUN:    -o - | FileCheck %s --check-prefix=CHECK-S

// Testing cases where we can elide the module initializer guard variable.

// This module has no global inits and does not import any other module
//--- O.cpp

export module O;

export int foo ();

// CHECK-O: define void @_ZGIW1O
// CHECK-O-LABEL: entry
// CHECK-O-NEXT: ret void
// CHECK-O-NOT: @_ZGIW1O__in_chrg

// This has no global inits and all the imported modules don't need inits. So
// guard variable is not needed.
//--- P.cpp

export module P;

export import O;
export int bar ();

// CHECK-P: define void @_ZGIW1P
// CHECK-P-LABEL: entry
// CHECK-P-NEXT: ret void
// CHECK-P-NOT: @_ZGIW1P__in_chrg

// This has global inits, so needs a guard.
//--- Q.cpp

export module Q;

export struct Quack {
  Quack(){};
};

export Quack Duck;

export int baz ();

// CHECK-Q: define internal void @__cxx_global_var_init
// CHECK-Q: call {{.*}} @_ZNW1Q5QuackC1Ev
// CHECK-Q: define void @_ZGIW1Q
// CHECK-Q: store i8 1, ptr @_ZGIW1Q__in_chrg
// CHECK-Q: call void @__cxx_global_var_init

// This doesn't have a global init, but it imports a module which needs global
// init, so needs a guard
//--- R.cpp

export module R;
export import Q;

// CHECK-R: define void @_ZGIW1R
// CHECK-R: store i8 1, ptr @_ZGIW1R__in_chrg
// CHECK-R: call{{.*}}@_ZGIW1Q

// This doesn't have a global init and the imported module doesn't have variables needs
// dynamic initialization.
// But the imported module contains modules initialization. So needs a guard.
//--- S.cpp

export module S;
export import R;

// CHECK-S: define void @_ZGIW1S
// CHECK-S: store i8 1, ptr @_ZGIW1S__in_chrg
// CHECK-S: call{{.*}}@_ZGIW1R
