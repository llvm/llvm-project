// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Declaration with an incomplete struct type.
struct U *p;

// CIR: cir.global external @p = #cir.ptr<null> : !cir.ptr<!cir.struct<struct "U" incomplete>>
// LLVM: @p = dso_local global ptr null
// OGCG: @p = global ptr null, align 8

void f(void) {
  struct U2 *p;
}

// CIR: cir.func @f()
// CIR-NEXT: cir.alloca !cir.ptr<!cir.struct<struct "U2" incomplete>>,
// CIR-SAME:     !cir.ptr<!cir.ptr<!cir.struct<struct "U2" incomplete>>>, ["p"]
// CIR-NEXT: cir.return
