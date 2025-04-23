// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct IncompleteS;
IncompleteS *p;

// CIR: cir.global external @p = #cir.ptr<null> : !cir.ptr<!cir.record<struct "IncompleteS" incomplete>>
// LLVM: @p = dso_local global ptr null
// OGCG: @p = global ptr null, align 8

struct IncompleteS* f(void) {
  IncompleteS *p;
  return p;
}

// CIR: cir.func @f()
// CIR:  cir.alloca !cir.ptr<!cir.record<struct "IncompleteS" incomplete>>, !cir.ptr<!cir.ptr<!cir.record<struct "IncompleteS" incomplete>>>, ["p"]

// LLVM: define ptr @f()
// LLVM:   %[[P:.*]] = alloca ptr, i64 1, align 8

// OGCG: define{{.*}} ptr @_Z1fv()
// OGCG-NEXT: entry:
// OGCG:   %[[P:.*]] = alloca ptr, align 8
