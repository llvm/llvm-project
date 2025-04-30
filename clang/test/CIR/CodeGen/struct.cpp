// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct IncompleteS;
IncompleteS *p;

// CIR: cir.global external @p = #cir.ptr<null> : !cir.ptr<!rec_IncompleteS>
// LLVM: @p = dso_local global ptr null
// OGCG: @p = global ptr null, align 8

void f(void) {
  IncompleteS *p;
}

// CIR:      cir.func @_Z1fv()
// CIR-NEXT:   cir.alloca !cir.ptr<!rec_IncompleteS>, !cir.ptr<!cir.ptr<!rec_IncompleteS>>, ["p"]
// CIR-NEXT:   cir.return

// LLVM:      define void @_Z1fv()
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @_Z1fv()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   ret void
