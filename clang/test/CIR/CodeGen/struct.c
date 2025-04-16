// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct IncompleteS *p;

// CIR: cir.global external @p = #cir.ptr<null> : !cir.ptr<!cir.record<struct "IncompleteS" incomplete>>
// LLVM: @p = dso_local global ptr null
// OGCG: @p = global ptr null, align 8

struct CompleteS {
  int a;
  char b;
} cs;

// CIR:      cir.global external @cs = #cir.zero : !cir.record<struct "CompleteS" {!s32i, !s8i}>
// LLVM:      @cs = dso_local global %struct.CompleteS zeroinitializer
// OGCG:      @cs = global %struct.CompleteS zeroinitializer, align 4

void f(void) {
  struct IncompleteS *p;
}

// CIR:      cir.func @f()
// CIR-NEXT:   cir.alloca !cir.ptr<!cir.record<struct "IncompleteS" incomplete>>,
// CIR-SAME:       !cir.ptr<!cir.ptr<!cir.record<struct "IncompleteS" incomplete>>>, ["p"]
// CIR-NEXT:   cir.return

// LLVM:      define void @f()
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @f()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   ret void

void f2(void) {
  struct CompleteS s;
}

// CIR:      cir.func @f2()
// CIR-NEXT:   cir.alloca !cir.record<struct "CompleteS" {!s32i, !s8i}>,
// CIR-SAME:       !cir.ptr<!cir.record<struct "CompleteS" {!s32i, !s8i}>>, ["s"]
// CIR-NEXT:   cir.return

// LLVM:      define void @f2()
// LLVM-NEXT:   %[[S:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @f2()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[S:.*]] = alloca %struct.CompleteS, align 4
// OGCG-NEXT:   ret void
