// RUN: rm -rf %t
// RUN: split-file %s %t
//
// Importing a clang header module emits the initializers of the module and of
// its implicit submodules into the importing translation unit, once.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -emit-cir %t/header-module.cpp -o - | FileCheck %t/header-module.cpp --check-prefix=CIR --implicit-check-not=c_global
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -emit-llvm %t/header-module.cpp -o - | FileCheck %t/header-module.cpp --check-prefix=LLVM --implicit-check-not=c_global
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -emit-llvm %t/header-module.cpp -o - | FileCheck %t/header-module.cpp --check-prefix=LLVM --implicit-check-not=c_global
//
// Importing a C++20 named module only records the module: its initializers
// stay in its own translation unit, and the importing unit's global init
// function calls the module's initializer first.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-cir %t/named-module.cpp -o - | FileCheck %t/named-module.cpp --check-prefix=CIR --implicit-check-not=__cxx_global_var_init
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-llvm %t/named-module.cpp -o - | FileCheck %t/named-module.cpp --check-prefix=LLVM --implicit-check-not=__cxx_global_var_init
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fmodule-file=a=%t/a.pcm -emit-llvm %t/named-module.cpp -o - | FileCheck %t/named-module.cpp --check-prefix=LLVM --implicit-check-not=__cxx_global_var_init
//
// The interface unit's own initializer, guarded; one of a module that only
// imports a module with an initializer, which calls it; one of a module with
// nothing to run, emitted all the same.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %t/a.cppm -o - | FileCheck %t/a.cppm --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %t/a.cppm -o - | FileCheck %t/a.cppm --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %t/a.cppm -o - | FileCheck %t/a.cppm --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-cir %t/b.cppm -o - | FileCheck %t/b.cppm --check-prefix=CIR --implicit-check-not=__cxx_global_var_init
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-llvm %t/b.cppm -o - | FileCheck %t/b.cppm --check-prefix=LLVM --implicit-check-not=__cxx_global_var_init
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fmodule-file=a=%t/a.pcm -emit-llvm %t/b.cppm -o - | FileCheck %t/b.cppm --check-prefix=LLVM --implicit-check-not=__cxx_global_var_init
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %t/c.cppm -o - | FileCheck %t/c.cppm --check-prefix=CIR --implicit-check-not=__in_chrg --implicit-check-not=cir.call
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %t/c.cppm -o - | FileCheck %t/c.cppm --check-prefix=LLVM --implicit-check-not=__in_chrg --implicit-check-not=call
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %t/c.cppm -o - | FileCheck %t/c.cppm --check-prefix=LLVM --implicit-check-not=__in_chrg --implicit-check-not=call
//
// init_priority: an importing unit calls the imported initializers from its
// first prioritized function; an interface unit folds its prioritized
// initializers into its own initializer, behind the guard.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-cir %t/named-module-priority.cpp -o - | FileCheck %t/named-module-priority.cpp --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-llvm %t/named-module-priority.cpp -o - | FileCheck %t/named-module-priority.cpp --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fmodule-file=a=%t/a.pcm -emit-llvm %t/named-module-priority.cpp -o - | FileCheck %t/named-module-priority.cpp --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-cir %t/d.cppm -o - | FileCheck %t/d.cppm --check-prefix=CIR --implicit-check-not=_GLOBAL__I_
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-llvm %t/d.cppm -o - | FileCheck %t/d.cppm --check-prefix=LLVM --implicit-check-not=_GLOBAL__I_
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fmodule-file=a=%t/a.pcm -emit-llvm %t/d.cppm -o - | FileCheck %t/d.cppm --check-prefix=LLVM --implicit-check-not=_GLOBAL__I_

//--- module.modulemap
module A {
  header "a.h"
  module B { header "b.h" }
  explicit module C { header "c.h" }
  export *
}

//--- a.h
int side_effect();
int a_global = side_effect();
inline int a_func(int x) { return x + 1; }

//--- b.h
int side_effect();
int b_global = side_effect();

//--- c.h
int side_effect();
int c_global = side_effect();

//--- header-module.cpp
// The second include of a modular header imports the module again and must not
// redefine its globals.
#include "a.h"
#include "a.h"

int use() { return a_func(a_global); }

// CIR: cir.global external {{.*}}@a_global = #cir.int<0> : !s32i
// CIR: cir.func internal private @__cxx_global_var_init()
// CIR:   cir.get_global @a_global : !cir.ptr<!s32i>
// CIR:   cir.call @_Z11side_effectv()
// CIR: cir.global external {{.*}}@b_global = #cir.int<0> : !s32i
// CIR: cir.func internal private @__cxx_global_var_init.1()
// CIR:   cir.get_global @b_global : !cir.ptr<!s32i>
// CIR:   cir.call @_Z11side_effectv()
// CIR-DAG: cir.func private @_Z11side_effectv()
// CIR-DAG: cir.func {{.*}}@_Z3usev()
// CIR-DAG: cir.func {{.*}}linkonce_odr @_Z6a_funci
// CIR: cir.func internal private @_GLOBAL__sub_I_header_module.cpp()
// CIR:   cir.call @__cxx_global_var_init()
// CIR:   cir.call @__cxx_global_var_init.1()

// LLVM: @a_global = global i32 0
// LLVM: @b_global = global i32 0
// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_header_module.cpp, ptr null }]
// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   call noundef i32 @_Z11side_effectv()
// LLVM:   store i32 %{{.*}}, ptr @a_global
// LLVM: define internal void @__cxx_global_var_init.1()
// LLVM:   call noundef i32 @_Z11side_effectv()
// LLVM:   store i32 %{{.*}}, ptr @b_global
// LLVM: define internal void @_GLOBAL__sub_I_header_module.cpp()
// LLVM:   call void @__cxx_global_var_init()
// LLVM:   call void @__cxx_global_var_init.1()

//--- a.cppm
export module a;
int side_effect();
export int a_val = side_effect();
export inline int a_func(int x) { return x + 1; }

// The initializer of a module interface unit has external linkage and runs
// its initializers once, behind a guard byte, whichever caller comes first.
// CIR: cir.cxx_module_init_fn_name = "_ZGIW1a"
// CIR: cir.global {{.*}}internal @_ZGIW1a__in_chrg = #cir.int<0> : !s8i
// CIR: cir.func private @_ZGIW1a()
// CIR:   cir.get_global @_ZGIW1a__in_chrg : !cir.ptr<!s8i>
// CIR:   cir.if
// CIR:     cir.store
// CIR:     cir.call @__cxx_global_var_init()

// LLVM: @_ZGIW1a__in_chrg = internal global i8 0
// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_ZGIW1a, ptr null }]
// LLVM: define void @_ZGIW1a()
// LLVM:   load i8, ptr @_ZGIW1a__in_chrg
// LLVM:   icmp eq i8
// LLVM:   store i8 1, ptr @_ZGIW1a__in_chrg
// LLVM:   call void @__cxx_global_var_init()

//--- named-module.cpp
import a;

int use() { return a_func(a_val); }

// CIR: cir.cxx_module_imported_inits = ["_ZGIW1a"]
// CIR: cir.global "private" external {{.*}}@_ZW1a5a_val : !s32i
// CIR: cir.func {{.*}}@_Z3usev()
// CIR:   cir.get_global @_ZW1a5a_val : !cir.ptr<!s32i>
// CIR:   cir.call @_ZW1a6a_funci
// CIR: cir.func {{.*}}linkonce_odr @_ZW1a6a_funci
// CIR: cir.func private @_ZGIW1a()
// CIR: cir.func internal private @_GLOBAL__sub_I_named_module.cpp()
// CIR:   cir.call @_ZGIW1a()

// LLVM: @_ZW1a5a_val = external global i32
// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_named_module.cpp, ptr null }]
// LLVM: define dso_local noundef i32 @_Z3usev()
// LLVM:   load i32, ptr @_ZW1a5a_val
// LLVM:   call noundef i32 @_ZW1a6a_funci
// LLVM: define linkonce_odr noundef i32 @_ZW1a6a_funci
// LLVM: declare void @_ZGIW1a()
// LLVM: define internal void @_GLOBAL__sub_I_named_module.cpp()
// LLVM:   call void @_ZGIW1a()

//--- b.cppm
export module b;
import a;
export int b_func() { return a_func(a_val); }

// CIR: cir.cxx_module_imported_inits = ["_ZGIW1a"]
// CIR: cir.cxx_module_init_fn_name = "_ZGIW1b"
// CIR: cir.global {{.*}}internal @_ZGIW1b__in_chrg = #cir.int<0> : !s8i
// CIR: cir.func private @_ZGIW1a()
// CIR: cir.func private @_ZGIW1b()
// CIR:   cir.get_global @_ZGIW1b__in_chrg : !cir.ptr<!s8i>
// CIR:   cir.if
// CIR:     cir.store
// CIR:     cir.call @_ZGIW1a()

// LLVM: @_ZGIW1b__in_chrg = internal global i8 0
// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_ZGIW1b, ptr null }]
// LLVM: declare void @_ZGIW1a()
// LLVM: define void @_ZGIW1b()
// LLVM:   load i8, ptr @_ZGIW1b__in_chrg
// LLVM:   store i8 1, ptr @_ZGIW1b__in_chrg
// LLVM:   call void @_ZGIW1a()

//--- c.cppm
export module c;
export int c_func();

// CIR: cir.func private @_ZGIW1c()
// CIR-NEXT:   cir.return

// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_ZGIW1c, ptr null }]
// LLVM: define void @_ZGIW1c()
// LLVM:   ret void

//--- named-module-priority.cpp
import a;
struct S { S(); };
int side_effect();
[[gnu::init_priority(200)]] S p;
int q = side_effect();

// The imported module's initializer runs from the first prioritized function,
// before that priority's own initializers; the default-priority function keeps
// the rest.
// CIR: cir.global_ctors = [#cir.global_ctor<"_GLOBAL__I_000200", 200>, #cir.global_ctor<"_GLOBAL__sub_I_named_module_priority.cpp", 65535>]
// CIR: cir.func internal private @_GLOBAL__I_000200()
// CIR-NEXT:   cir.call @_ZGIW1a()
// CIR-NEXT:   cir.call @__cxx_global_var_init()
// CIR-NEXT:   cir.return
// CIR: cir.func internal private @_GLOBAL__sub_I_named_module_priority.cpp()
// CIR-NEXT:   cir.call @__cxx_global_var_init.1()
// CIR-NEXT:   cir.return

// LLVM: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 200, ptr @_GLOBAL__I_000200, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_named_module_priority.cpp, ptr null }]
// LLVM: define internal void @_GLOBAL__I_000200()
// LLVM:   call void @_ZGIW1a()
// LLVM-NEXT:   call void @__cxx_global_var_init()
// LLVM: define internal void @_GLOBAL__sub_I_named_module_priority.cpp()
// LLVM:   call void @__cxx_global_var_init.1()

//--- d.cppm
export module d;
import a;
struct S { S(); };
int side_effect();
[[gnu::init_priority(200)]] S p;
int q = side_effect();

// An interface unit folds its prioritized initializers into its own
// initializer, after the imported modules' and behind the guard.
// CIR: cir.global_ctors = [#cir.global_ctor<"_ZGIW1d", 65535>]
// CIR: cir.func private @_ZGIW1d()
// CIR:   cir.if
// CIR:     cir.store
// CIR-NEXT:     cir.call @_ZGIW1a()
// CIR-NEXT:     cir.call @__cxx_global_var_init()
// CIR-NEXT:     cir.call @__cxx_global_var_init.1()

// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_ZGIW1d, ptr null }]
// LLVM: define void @_ZGIW1d()
// LLVM:   store i8 1, ptr @_ZGIW1d__in_chrg
// LLVM-NEXT:   call void @_ZGIW1a()
// LLVM-NEXT:   call void @__cxx_global_var_init()
// LLVM-NEXT:   call void @__cxx_global_var_init.1()
