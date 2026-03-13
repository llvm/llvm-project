// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

extern __thread int b;
// CIR: cir.global "private" external tls_dyn @b : !s32i

__thread int a;
// CIR: cir.global external tls_dyn @a = #cir.int<0> : !s32i

int c(void) { return *&b; }
// CIR: cir.func no_inline dso_local @c() -> !s32i
// CIR:   %[[TLS_ADDR:.*]] = cir.get_global thread_local @b : !cir.ptr<!s32i>

// LLVM: @b = external thread_local global i32
// LLVM: @a = thread_local global i32 0

// LLVM-LABEL: @c
// LLVM: = call ptr @llvm.threadlocal.address.p0(ptr @b)

// OGCG: @b = external thread_local{{.*}} global i32
// OGCG: @a = thread_local{{.*}} global i32 0

// OGCG-LABEL: define{{.*}} @c
// OGCG: call{{.*}} ptr @llvm.threadlocal.address.p0(ptr{{.*}} @b)

