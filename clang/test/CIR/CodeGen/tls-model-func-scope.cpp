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

int init();

void func() {
  thread_local int default_tls_mode = init();

  __attribute__((tls_model("initial-exec")))
  thread_local int override_tls_mode = init();
}
// CIR-GD: module {{.*}} attributes
// CIR-GD-SAME: cir.default_tls_model = #cir.tls_model<tls_dyn>
// CIR-GD: cir.global "private" internal tls_model = <tls_init_exec> dso_local @_ZGVZ4funcvE17override_tls_mode = #cir.int<0> : !s8i
// CIR-GD: cir.global "private" internal tls_model = <tls_dyn> dso_local @_ZGVZ4funcvE16default_tls_mode = #cir.int<0> : !s8i
// CIR-GD: cir.global "private" internal tls_model = <tls_init_exec> dso_local static_local_guard<"_ZGVZ4funcvE17override_tls_mode"> @_ZZ4funcvE17override_tls_mode = #cir.int<0> : !s32i
// CIR-GD: cir.global "private" internal tls_model = <tls_dyn> dso_local static_local_guard<"_ZGVZ4funcvE16default_tls_mode"> @_ZZ4funcvE16default_tls_mode = #cir.int<0> : !s32i

// LLVM-GD-DAG: @_ZGVZ4funcvE17override_tls_mode = internal thread_local(initialexec) global i8 0
// LLVM-GD-DAG: @_ZGVZ4funcvE16default_tls_mode = internal thread_local global i8 0
// LLVM-GD-DAG: @_ZZ4funcvE17override_tls_mode = internal thread_local(initialexec) global i32 0
// LLVM-GD-DAG: @_ZZ4funcvE16default_tls_mode = internal thread_local global i32 0

// CIR-LD: module {{.*}} attributes
// CIR-LD-SAME: cir.default_tls_model = #cir.tls_model<tls_local_dyn>
// CIR-LD: cir.global "private" internal tls_model = <tls_init_exec> dso_local @_ZGVZ4funcvE17override_tls_mode = #cir.int<0> : !s8i
// CIR-LD: cir.global "private" internal tls_model = <tls_local_dyn> dso_local @_ZGVZ4funcvE16default_tls_mode = #cir.int<0> : !s8i
// CIR-LD: cir.global "private" internal tls_model = <tls_init_exec> dso_local static_local_guard<"_ZGVZ4funcvE17override_tls_mode"> @_ZZ4funcvE17override_tls_mode = #cir.int<0> : !s32i
// CIR-LD: cir.global "private" internal tls_model = <tls_local_dyn> dso_local static_local_guard<"_ZGVZ4funcvE16default_tls_mode"> @_ZZ4funcvE16default_tls_mode = #cir.int<0> : !s32i

// LLVM-LD-DAG: @_ZGVZ4funcvE17override_tls_mode = internal thread_local(initialexec) global i8 0
// LLVM-LD-DAG: @_ZGVZ4funcvE16default_tls_mode = internal thread_local(localdynamic) global i8 0
// LLVM-LD-DAG: @_ZZ4funcvE17override_tls_mode = internal thread_local(initialexec) global i32 0
// LLVM-LD-DAG: @_ZZ4funcvE16default_tls_mode = internal thread_local(localdynamic) global i32 0

// CIR-IE: module {{.*}} attributes
// CIR-IE-SAME: cir.default_tls_model = #cir.tls_model<tls_init_exec>
// CIR-IE: cir.global "private" internal tls_model = <tls_init_exec> dso_local @_ZGVZ4funcvE17override_tls_mode = #cir.int<0> : !s8i
// CIR-IE: cir.global "private" internal tls_model = <tls_init_exec> dso_local @_ZGVZ4funcvE16default_tls_mode = #cir.int<0> : !s8i
// CIR-IE: cir.global "private" internal tls_model = <tls_init_exec> dso_local static_local_guard<"_ZGVZ4funcvE17override_tls_mode"> @_ZZ4funcvE17override_tls_mode = #cir.int<0> : !s32i
// CIR-IE: cir.global "private" internal tls_model = <tls_init_exec> dso_local static_local_guard<"_ZGVZ4funcvE16default_tls_mode"> @_ZZ4funcvE16default_tls_mode = #cir.int<0> : !s32i

// LLVM-IE-DAG: @_ZGVZ4funcvE17override_tls_mode = internal thread_local(initialexec) global i8 0
// LLVM-IE-DAG: @_ZGVZ4funcvE16default_tls_mode = internal thread_local(initialexec) global i8 0
// LLVM-IE-DAG: @_ZZ4funcvE17override_tls_mode = internal thread_local(initialexec) global i32 0
// LLVM-IE-DAG: @_ZZ4funcvE16default_tls_mode = internal thread_local(initialexec) global i32 0

// CIR-LE: module {{.*}} attributes
// CIR-LE-SAME: cir.default_tls_model = #cir.tls_model<tls_local_exec>
// CIR-LE: cir.global "private" internal tls_model = <tls_init_exec> dso_local @_ZGVZ4funcvE17override_tls_mode = #cir.int<0> : !s8i
// CIR-LE: cir.global "private" internal tls_model = <tls_local_exec> dso_local @_ZGVZ4funcvE16default_tls_mode = #cir.int<0> : !s8i
// CIR-LE: cir.global "private" internal tls_model = <tls_init_exec> dso_local static_local_guard<"_ZGVZ4funcvE17override_tls_mode"> @_ZZ4funcvE17override_tls_mode = #cir.int<0> : !s32i
// CIR-LE: cir.global "private" internal tls_model = <tls_local_exec> dso_local static_local_guard<"_ZGVZ4funcvE16default_tls_mode"> @_ZZ4funcvE16default_tls_mode = #cir.int<0> : !s32i

// LLVM-LE-DAG: @_ZGVZ4funcvE17override_tls_mode = internal thread_local(initialexec) global i8 0
// LLVM-LE-DAG: @_ZGVZ4funcvE16default_tls_mode = internal thread_local(localexec) global i8 0
// LLVM-LE-DAG: @_ZZ4funcvE17override_tls_mode = internal thread_local(initialexec) global i32 0
// LLVM-LE-DAG: @_ZZ4funcvE16default_tls_mode = internal thread_local(localexec) global i32 0
