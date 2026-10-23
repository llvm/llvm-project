// UNSUPPORTED: target={{.*}}-apple-darwin

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 NoInit.cpp \
// RUN:   -emit-module-interface -o NoInit.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 NoInit.pcm \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=NO-INIT-MODULE
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 UseNoInit.cpp \
// RUN:   -fprebuilt-module-path=%t -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefix=NO-INIT
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 HasDtor.cpp \
// RUN:   -emit-module-interface -o HasDtor.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 UseHasDtor.cpp \
// RUN:   -fprebuilt-module-path=%t -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefix=HAS-DTOR
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 HasDynamic.cpp \
// RUN:   -emit-module-interface -o HasDynamic.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 UseHasDynamic.cpp \
// RUN:   -fprebuilt-module-path=%t -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefix=HAS-DYNAMIC

// Test again for reduced BMI
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 NoInit.cpp \
// RUN:   -emit-reduced-module-interface -o NoInit.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 NoInit.pcm \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=NO-INIT-MODULE
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 UseNoInit.cpp \
// RUN:   -fprebuilt-module-path=%t -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefix=NO-INIT
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 HasDtor.cpp \
// RUN:   -emit-reduced-module-interface -o HasDtor.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 UseHasDtor.cpp \
// RUN:   -fprebuilt-module-path=%t -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefix=HAS-DTOR
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 HasDynamic.cpp \
// RUN:   -emit-reduced-module-interface -o HasDynamic.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 UseHasDynamic.cpp \
// RUN:   -fprebuilt-module-path=%t -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefix=HAS-DYNAMIC

//--- NoInit.cpp
export module NoInit;

// Neither constant initialization nor dynamic TLS initialization contributes
// to the module initializer.
int a = 43;
int f();
thread_local int tls = f();

// NO-INIT-MODULE-LABEL: define void @_ZGIW6NoInit()
// NO-INIT-MODULE: entry:
// NO-INIT-MODULE-NEXT: ret void

//--- UseNoInit.cpp
import NoInit;

// NO-INIT: source_filename = {{.*}}UseNoInit.cpp
// NO-INIT-NOT: @_ZGIW6NoInit
// NO-INIT-NOT: @llvm.global_ctors

//--- HasDtor.cpp
export module HasDtor;

// Constant initialization still needs a module initializer when it registers a
// destructor.
struct S {
  constexpr S() = default;
  ~S() {}
};
constinit S s;

//--- UseHasDtor.cpp
import HasDtor;

// HAS-DTOR: define internal void @_GLOBAL__sub_I_UseHasDtor.cpp()
// HAS-DTOR: call void @_ZGIW7HasDtor()

//--- HasDynamic.cpp
export module HasDynamic;

// Dynamic initialization contributes to the module initializer.
int f();
int n = f();

//--- UseHasDynamic.cpp
import HasDynamic;

// HAS-DYNAMIC: define internal void @_GLOBAL__sub_I_UseHasDynamic.cpp()
// HAS-DYNAMIC: call void @_ZGIW10HasDynamic()
