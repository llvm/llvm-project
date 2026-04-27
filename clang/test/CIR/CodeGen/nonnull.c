// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// __attribute__((nonnull)) on individual parameter.
void take_nonnull(int *p) __attribute__((nonnull(1)));
void take_nonnull(int *p) { *p = 42; }
// CIR: cir.func {{.*}} @take_nonnull(%arg0: !cir.ptr<!s32i>
// CIR-SAME: {llvm.nonnull, llvm.noundef}
// LLVM: define {{.*}} void @take_nonnull(ptr noundef nonnull %{{.*}})
// OGCG: define {{.*}} void @take_nonnull(ptr noundef nonnull %{{.*}})

// Function-level nonnull applying to all pointer params.
__attribute__((nonnull)) void take_all_nonnull(int *a, int *b);
void take_all_nonnull(int *a, int *b) { *a = *b; }
// CIR: cir.func {{.*}} @take_all_nonnull(
// CIR-SAME: %arg0: !cir.ptr<!s32i> {llvm.nonnull, llvm.noundef}
// CIR-SAME: %arg1: !cir.ptr<!s32i> {llvm.nonnull, llvm.noundef}
// LLVM: define {{.*}} void @take_all_nonnull(ptr noundef nonnull %{{.*}}, ptr noundef nonnull %{{.*}})
// OGCG: define {{.*}} void @take_all_nonnull(ptr noundef nonnull %{{.*}}, ptr noundef nonnull %{{.*}})
