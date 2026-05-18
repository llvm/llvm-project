// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

const char *names[] = { "a", "b", "c" };
int len() { return sizeof(names)/sizeof(*names); }

// CIR: cir.global {{.*}}@names = #cir.const_array<[#cir.global_view<@".str"> : !cir.ptr<!s8i>, #cir.global_view<@".str.1"> : !cir.ptr<!s8i>, #cir.global_view<@".str.2"> : !cir.ptr<!s8i>]>

// LLVM:       @names = {{.*}}global [3 x ptr] [ptr @.str{{.*}}, ptr @.str{{.*}}, ptr @.str{{.*}}]
// LLVM-NOT:   insertvalue

// OGCG:       @names = {{.*}}global [3 x ptr] [ptr @.str{{.*}}, ptr @.str{{.*}}, ptr @.str{{.*}}]
