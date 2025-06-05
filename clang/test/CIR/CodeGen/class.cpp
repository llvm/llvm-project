// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// CIR: !rec_IncompleteC = !cir.record<class "IncompleteC" incomplete>
// CIR: !rec_CompleteC = !cir.record<class "CompleteC" {!s32i, !s8i}>

// Note: LLVM and OGCG do not emit the type for incomplete classes.

// LLVM: %class.CompleteC = type { i32, i8 }

// OGCG: %class.CompleteC = type { i32, i8 }

class IncompleteC;
IncompleteC *p;

// CIR: cir.global external @p = #cir.ptr<null> : !cir.ptr<!rec_IncompleteC>
// LLVM: @p = global ptr null
// OGCG: @p = global ptr null, align 8

class CompleteC {
public:    
  int a;
  char b;
};

CompleteC cc;

// CIR:       cir.global external @cc = #cir.zero : !rec_CompleteC
// LLVM:  @cc = global %class.CompleteC zeroinitializer
// OGCG:  @cc = global %class.CompleteC zeroinitializer
