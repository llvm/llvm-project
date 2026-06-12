// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void a(void) {
A:;
  static void *a = &&A;
}

// CIR: cir.global "private" internal dso_local @a.a = #cir.block_addr_info<@a, "A"> : !cir.ptr<!void>
// CIR: cir.func{{.*}} @a()
// CIR:   cir.br ^[[A_BLOCK:bb[0-9]+]]
// CIR: ^[[A_BLOCK]]:
// CIR:   cir.label "A"
// CIR:   %[[STATIC_A:.*]] = cir.get_global @a.a : !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.return

// LLVM: @a.a = internal global ptr blockaddress(@a, %[[A_BLOCK:.*]]), align 8
// LLVM: define dso_local void @a()
// LLVM:   br label %[[A_BLOCK]]
// LLVM: [[A_BLOCK]]:
// LLVM:   ret void
