// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -emit-cir -o - | FileCheck %s -check-prefix=CIR-GD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -emit-llvm -fclangir -o - | FileCheck %s -check-prefix=LLVM-GD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -emit-llvm -o - | FileCheck %s -check-prefix=LLVM-GD

// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=global-dynamic -emit-cir -o - | FileCheck %s -check-prefix=CIR-GD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=global-dynamic -emit-llvm -fclangir -o - | FileCheck %s -check-prefix=LLVM-GD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=global-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=LLVM-GD

// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=local-dynamic -emit-cir -o - | FileCheck %s -check-prefix=CIR-LD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=local-dynamic -emit-llvm -fclangir -o - | FileCheck %s -check-prefix=LLVM-LD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=local-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=LLVM-LD

// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=initial-exec -emit-cir -o - | FileCheck %s -check-prefix=CIR-IE
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=initial-exec -emit-llvm -fclangir -o - | FileCheck %s -check-prefix=LLVM-IE
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=initial-exec -emit-llvm -o - | FileCheck %s -check-prefix=LLVM-IE

// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=local-exec -emit-cir -o - | FileCheck %s -check-prefix=CIR-LE
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=local-exec -emit-llvm -fclangir -o - | FileCheck %s -check-prefix=LLVM-LE
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=local-exec -emit-llvm -o - | FileCheck %s -check-prefix=LLVM-LE

// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -femulated-tls -emit-cir -o - 2>&1 | FileCheck %s -check-prefix=CIR-GD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -femulated-tls -emit-llvm -fclangir -o - 2>&1 | FileCheck %s -check-prefix=LLVM-GD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -femulated-tls -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=LLVM-GD

int z1 = 0;
int z2;
int __thread x;
int f() {
  static int __thread y;
  return y++;
}
int __thread __attribute__((tls_model("initial-exec"))) z;

struct S {
  S();
  ~S();
};
struct T {
  ~T();
};

struct S thread_local s1;
struct T thread_local t1;

// Note that unlike normal C uninitialized global variables,
// uninitialized TLS variables do NOT have COMMON linkage.

// CIR-GD: module {{.*}} attributes 
// CIR-GD-SAME: cir.default_tls_model = #cir.tls_model<tls_dyn>
// CIR-GD-DAG: cir.global external @z1 = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-GD-DAG: cir.global external @z2 = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-GD-DAG: cir.global external tls_model = <tls_dyn> @x = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-GD-DAG: cir.global "private" internal tls_model = <tls_dyn> dso_local @_ZZ1fvE1y = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-GD-DAG: cir.global external tls_model = <tls_init_exec> @z = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-GD-DAG: cir.global external tls_model = <tls_dyn> tls_refs = <"_ZTW2s1", "_ZTH2s1"> @s1 = #cir.zero : !rec_S {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-GD-DAG: cir.global external tls_model = <tls_dyn> tls_refs = <"_ZTW2t1", "_ZTH2t1"> @t1 = #cir.zero : !rec_T {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-GD-DAG: cir.global internal tls_model = <tls_dyn> @__tls_guard = #cir.int<0> : !s8i {alignment = 1 : i64}

// LLVM-GD-DAG: @z1 ={{.*}} global i32 0
// LLVM-GD-DAG: @z2 ={{.*}} global i32 0
// LLVM-GD-DAG: @x ={{.*}} thread_local global i32 0
// LLVM-GD-DAG: @_ZZ1fvE1y = internal thread_local global i32 0
// LLVM-GD-DAG: @z ={{.*}} thread_local(initialexec) global i32 0
// LLVM-GD-DAG: @s1 ={{.*}} thread_local global %struct.S zeroinitializer
// LLVM-GD-DAG: define weak_odr hidden {{.*}}ptr @_ZTW2s1()
// LLVM-GD-DAG: @_ZTH2s1 = alias void (), ptr @__tls_init
// LLVM-GD-DAG: @t1 ={{.*}} thread_local global %struct.T zeroinitializer
// LLVM-GD-DAG: define weak_odr hidden {{.*}}ptr @_ZTW2t1()
// LLVM-GD-DAG: @_ZTH2t1 = alias void (), ptr @__tls_init
// LLVM-GD-DAG: @__tls_guard = internal thread_local global i8 0

// CIR-LD: module {{.*}} attributes 
// CIR-LD-SAME: cir.default_tls_model = #cir.tls_model<tls_local_dyn>
// CIR-LD-DAG: cir.global external @z1 = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LD-DAG: cir.global external @z2 = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LD-DAG: cir.global external tls_model = <tls_local_dyn> @x = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LD-DAG: cir.global "private" internal tls_model = <tls_local_dyn> dso_local @_ZZ1fvE1y = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LD-DAG: cir.global external tls_model = <tls_init_exec> @z = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LD-DAG: cir.global external tls_model = <tls_local_dyn> tls_refs = <"_ZTW2s1", "_ZTH2s1"> @s1 = #cir.zero : !rec_S {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-LD-DAG: cir.global external tls_model = <tls_local_dyn> tls_refs = <"_ZTW2t1", "_ZTH2t1"> @t1 = #cir.zero : !rec_T {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-LD-DAG: cir.global internal tls_model = <tls_local_dyn> @__tls_guard = #cir.int<0> : !s8i {alignment = 1 : i64}

// LLVM-LD-DAG: @z1 ={{.*}} global i32 0
// LLVM-LD-DAG: @z2 ={{.*}} global i32 0
// LLVM-LD-DAG: @x ={{.*}} thread_local(localdynamic) global i32 0
// LLVM-LD-DAG: @_ZZ1fvE1y = internal thread_local(localdynamic) global i32 0
// LLVM-LD-DAG: @z ={{.*}} thread_local(initialexec) global i32 0
// LLVM-LD-DAG: @s1 ={{.*}} thread_local(localdynamic) global %struct.S zeroinitializer
// LLVM-LD-DAG: define weak_odr hidden {{.*}}ptr @_ZTW2s1()
// LLVM-LD-DAG: @_ZTH2s1 = alias void (), ptr @__tls_init
// LLVM-LD-DAG: @t1 ={{.*}} thread_local(localdynamic) global %struct.T zeroinitializer
// LLVM-LD-DAG: define weak_odr hidden {{.*}}ptr @_ZTW2t1()
// LLVM-LD-DAG: @_ZTH2t1 = alias void (), ptr @__tls_init
// LLVM-LD-DAG: @__tls_guard = internal thread_local(localdynamic) global i8 0

// CIR-IE: module {{.*}} attributes 
// CIR-IE-SAME: cir.default_tls_model = #cir.tls_model<tls_init_exec>
// CIR-IE-DAG: cir.global external @z1 = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-IE-DAG: cir.global external @z2 = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-IE-DAG: cir.global external tls_model = <tls_init_exec> @x = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-IE-DAG: cir.global "private" internal tls_model = <tls_init_exec> dso_local @_ZZ1fvE1y = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-IE-DAG: cir.global external tls_model = <tls_init_exec> @z = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-IE-DAG: cir.global external tls_model = <tls_init_exec> tls_refs = <"_ZTW2s1", "_ZTH2s1"> @s1 = #cir.zero : !rec_S {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-IE-DAG: cir.global external tls_model = <tls_init_exec> tls_refs = <"_ZTW2t1", "_ZTH2t1"> @t1 = #cir.zero : !rec_T {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-IE-DAG: cir.global internal tls_model = <tls_init_exec> @__tls_guard = #cir.int<0> : !s8i {alignment = 1 : i64}

// LLVM-IE-DAG: @z1 ={{.*}} global i32 0
// LLVM-IE-DAG: @z2 ={{.*}} global i32 0
// LLVM-IE-DAG: @x ={{.*}} thread_local(initialexec) global i32 0
// LLVM-IE-DAG: @_ZZ1fvE1y = internal thread_local(initialexec) global i32 0
// LLVM-IE-DAG: @z ={{.*}} thread_local(initialexec) global i32 0
// LLVM-IE-DAG: @s1 ={{.*}} thread_local(initialexec) global %struct.S zeroinitializer
// LLVM-IE-DAG: define weak_odr hidden {{.*}}ptr @_ZTW2s1()
// LLVM-IE-DAG: @_ZTH2s1 = alias void (), ptr @__tls_init
// LLVM-IE-DAG: @t1 ={{.*}} thread_local(initialexec) global %struct.T zeroinitializer
// LLVM-IE-DAG: define weak_odr hidden {{.*}}ptr @_ZTW2t1()
// LLVM-IE-DAG: @_ZTH2t1 = alias void (), ptr @__tls_init
// LLVM-IE-DAG: @__tls_guard = internal thread_local(initialexec) global i8 0

// CIR-LE: module {{.*}} attributes 
// CIR-LE-SAME: cir.default_tls_model = #cir.tls_model<tls_local_exec>
// CIR-LE-DAG: cir.global external @z1 = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LE-DAG: cir.global external @z2 = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LE-DAG: cir.global external tls_model = <tls_local_exec> @x = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LE-DAG: cir.global "private" internal tls_model = <tls_local_exec> dso_local @_ZZ1fvE1y = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LE-DAG: cir.global external tls_model = <tls_init_exec> @z = #cir.int<0> : !s32i {alignment = 4 : i64}
// CIR-LE-DAG: cir.global external tls_model = <tls_local_exec> tls_refs = <"_ZTW2s1", "_ZTH2s1"> @s1 = #cir.zero : !rec_S {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-LE-DAG: cir.global external tls_model = <tls_local_exec> tls_refs = <"_ZTW2t1", "_ZTH2t1"> @t1 = #cir.zero : !rec_T {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-LE-DAG: cir.global internal tls_model = <tls_local_exec> @__tls_guard = #cir.int<0> : !s8i {alignment = 1 : i64}

// LLVM-LE-DAG: @z1 ={{.*}} global i32 0
// LLVM-LE-DAG: @z2 ={{.*}} global i32 0
// LLVM-LE-DAG: @x ={{.*}} thread_local(localexec) global i32 0
// LLVM-LE-DAG: @_ZZ1fvE1y = internal thread_local(localexec) global i32 0
// LLVM-LE-DAG: @z ={{.*}} thread_local(initialexec) global i32 0
// LLVM-LE-DAG: @s1 ={{.*}} thread_local(localexec) global %struct.S zeroinitializer
// LLVM-LE-DAG: define weak_odr hidden {{.*}}ptr @_ZTW2s1()
// LLVM-LE-DAG: @_ZTH2s1 = alias void (), ptr @__tls_init
// LLVM-LE-DAG: @t1 ={{.*}} thread_local(localexec) global %struct.T zeroinitializer
// LLVM-LE-DAG: define weak_odr hidden {{.*}}ptr @_ZTW2t1()
// LLVM-LE-DAG: @_ZTH2t1 = alias void (), ptr @__tls_init
// LLVM-LE-DAG: @__tls_guard = internal thread_local(localexec) global i8 0
