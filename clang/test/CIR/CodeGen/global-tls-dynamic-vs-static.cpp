// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR,NOWRAP
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM,NOWRAP
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM,NOWRAP

int f();
__thread      int a = 5;
_Thread_local int b = 5;
thread_local  int c = 5;
thread_local  int d = f();

// CIR-BEFORE-LPP: cir.global external tls_model = tls_dyn @a = #cir.int<5> : !s32i
// CIR-BEFORE-LPP: cir.global external tls_model = tls_dyn @b = #cir.int<5> : !s32i
// CIR-BEFORE-LPP: cir.global external tls_model = tls_dyn tls_refs = <"_ZTW1c", "_ZTH1c"> @c = #cir.int<5> : !s32i
// CIR-BEFORE-LPP: cir.global external tls_model = tls_dyn tls_refs = <"_ZTW1d", "_ZTH1d"> @d = ctor

// CIR-DAG: cir.global external tls_model = tls_dyn @a = #cir.int<5> : !s32i
// CIR-DAG: cir.global external tls_model = tls_dyn @b = #cir.int<5> : !s32i

// CIR-DAG: cir.global internal tls_model = tls_dyn @__tls_guard = #cir.int<0> : !s8i
// CIR-DAG: cir.func internal private @__tls_init()
// CIR-DAG: cir.func internal private @__cxx_global_var_init()

// CIR-DAG: cir.global external tls_model = tls_dyn tls_refs = <"_ZTW1c", "_ZTH1c"> @c = #cir.int<5> : !s32i
// CIR-DAG: cir.func comdat weak_odr private hidden @_ZTW1c()
// Note: C doesn't get an alias since it doesn't have an init function.

// CIR-DAG: cir.global external tls_model = tls_dyn tls_refs = <"_ZTW1d", "_ZTH1d"> @d = #cir.int<0> : !s32i
// CIR-DAG: cir.func comdat weak_odr private hidden @_ZTW1d() -> !cir.ptr<!s32i>
// CIR-DAG: cir.func @_ZTH1d() alias(@__tls_init)

// LLVM-DAG: @__tls_guard = internal thread_local global i8 0
// LLVM-DAG: define internal void @__tls_init()
// LLVM-DAG: define internal void @__cxx_global_var_init()

// LLVM-DAG: @a = thread_local global i32 5
// LLVM-DAG: @b = thread_local global i32 5
// LLVM-DAG: @c = thread_local global i32 5
// LLVM-DAG: define weak_odr hidden {{.*}}ptr @_ZTW1c()

// LLVM-DAG: @d = thread_local global i32 0
// LLVM-DAG: @_ZTH1d = alias void (), ptr @__tls_init
// LLVM-DAG: define weak_odr hidden {{.*}}ptr @_ZTW1d()

// NOWRAP-NOT: @_ZTW1a
// NOWRAP-NOT: @_ZTH1a
// NOWRAP-NOT: @_ZTW1b
// NOWRAP-NOT: @_ZTH1b
// NOWRAP-NOT: @_ZTH1c
