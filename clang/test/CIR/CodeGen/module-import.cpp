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
// stay in its own translation unit. The call to that module's initializer from
// the importing unit's global init function is not emitted yet.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-cir %t/named-module.cpp -o - | FileCheck %t/named-module.cpp --check-prefix=CIR --implicit-check-not=__cxx_global_var_init
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fmodule-file=a=%t/a.pcm -emit-llvm %t/named-module.cpp -o - | FileCheck %t/named-module.cpp --check-prefix=LLVM --implicit-check-not=__cxx_global_var_init

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

//--- named-module.cpp
import a;

int use() { return a_func(a_val); }

// CIR: cir.global "private" external {{.*}}@_ZW1a5a_val : !s32i
// CIR: cir.func {{.*}}@_Z3usev()
// CIR:   cir.get_global @_ZW1a5a_val : !cir.ptr<!s32i>
// CIR:   cir.call @_ZW1a6a_funci
// CIR: cir.func {{.*}}linkonce_odr @_ZW1a6a_funci

// LLVM: @_ZW1a5a_val = external global i32
// LLVM: define dso_local noundef i32 @_Z3usev()
// LLVM:   load i32, ptr @_ZW1a5a_val
// LLVM:   call noundef i32 @_ZW1a6a_funci
// LLVM: define linkonce_odr noundef i32 @_ZW1a6a_funci
