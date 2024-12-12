// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-none.cir
// RUN: FileCheck %s --input-file=%t-none.cir --check-prefix=CIR-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -funwind-tables=0 %s -o %t-none-explicit.cir
// RUN: FileCheck %s --input-file=%t-none-explicit.cir --check-prefix=CIR-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -funwind-tables=1 %s -o %t-sync.cir
// RUN: FileCheck %s --input-file=%t-sync.cir --check-prefix=CIR-SYNC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -funwind-tables=2 %s -o %t-async.cir
// RUN: FileCheck %s --input-file=%t-async.cir --check-prefix=CIR-ASYNC

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-none.ll
// RUN: FileCheck %s --input-file=%t-none.ll --check-prefix=LLVM-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -funwind-tables=0 %s -o %t-none-explicit.ll
// RUN: FileCheck %s --input-file=%t-none-explicit.ll --check-prefix=LLVM-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -funwind-tables=1 %s -o %t-sync.ll
// RUN: FileCheck %s --input-file=%t-sync.ll --check-prefix=LLVM-SYNC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -funwind-tables=2 %s -o %t-async.ll
// RUN: FileCheck %s --input-file=%t-async.ll --check-prefix=LLVM-ASYNC

// CIR-NONE-NOT: #cir.uwtable

// CIR-SYNC-DAG: module {{.*}} attributes {{{.*}}cir.uwtable = #cir.uwtable<sync>
// CIR-SYNC-DAG:   cir.func @_Z1fv() extra(#[[f_attr:.*]])
// CIR-SYNC-DAG:   cir.func @_Z1gv() extra(#[[g_attr:.*]])
// CIR-SYNC-DAG: #[[f_attr]] = #cir<extra({{{.*}}uwtable = #cir.uwtable<sync>
// CIR-SYNC-DAG: #[[g_attr]] =
// CIR-SYNC-NOT:   #cir.uwtable

// CIR-ASYNC-DAG: module {{.*}} attributes {{{.*}}cir.uwtable = #cir.uwtable<async>
// CIR-ASYNC-DAG:   cir.func @_Z1fv() extra(#[[f_attr:.*]])
// CIR-ASYNC-DAG:   cir.func @_Z1gv() extra(#[[g_attr:.*]])
// CIR-ASYNC-DAG: #[[f_attr]] = #cir<extra({{{.*}}uwtable = #cir.uwtable<async>
// CIR-ASYNC-DAG: #[[g_attr]] =
// CIR-ASYNC-NOT:   #cir.uwtable

// Avoid matching "uwtable" in the ModuleID and source_filename comments.
// LLVM-NONE:     define {{.*}} @_Z1fv()
// LLVM-NONE-NOT: uwtable

// LLVM-SYNC:     define {{.*}} @_Z1fv() #[[#F_ATTRS:]]
// LLVM-SYNC:     define {{.*}} @_Z1gv() #[[#G_ATTRS:]]
// LLVM-SYNC:     attributes #[[#F_ATTRS]] = {{{.*}}uwtable(sync)
// LLVM-SYNC:     attributes #[[#G_ATTRS]] =
// LLVM-SYNC-NOT:   uwtable
// LLVM-SYNC-DAG: ![[#METADATA:]] = !{i32 7, !"uwtable", i32 1}
// LLVM-SYNC-DAG: !llvm.module.flags = !{{{.*}}[[#METADATA]]

// LLVM-ASYNC:     define {{.*}} @_Z1fv() #[[#ATTRS:]]
// LLVM-ASYNC:     define {{.*}} @_Z1gv() #[[#G_ATTRS:]]
// LLVM-ASYNC:     attributes #[[#ATTRS]] = {{{.*}}uwtable{{ }}
// LLVM-ASYNC:     attributes #[[#G_ATTRS]] =
// LLVM-ASYNC-NOT:   uwtable
// LLVM-ASYNC-DAG: ![[#METADATA:]] = !{i32 7, !"uwtable", i32 2}
// LLVM-ASYNC-DAG: !llvm.module.flags = !{{{.*}}[[#METADATA]]
void f() {}

[[clang::nouwtable]] void g() {}
