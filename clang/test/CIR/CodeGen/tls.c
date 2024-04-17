// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

extern __thread int b;
int c(void) { return *&b; }
// CIR: cir.global "private" external tls_dyn @b : !s32i
// CIR: cir.func @c() -> !s32i
// CIR:   %[[TLS_ADDR:.*]] = cir.get_global thread_local @b : cir.ptr <!s32i>

__thread int a;
// CIR: cir.global external tls_dyn @a = #cir.int<0> : !s32i

// LLVM: @b = external thread_local global i32
// LLVM: @a = thread_local global i32 0

// LLVM-LABEL: @c
// LLVM: = call ptr @llvm.threadlocal.address.p0(ptr @b)