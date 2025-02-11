// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -relaxed-aliasing
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O0
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s

// NO-TBAA-NOT: !tbaa
// CIR: #tbaa[[CHAR:.*]] = #cir.tbaa_omnipotent_char
typedef struct {
  union {
    int a, b;
  };
  int c;
} S;

void foo(S *s) {
  // CIR-LABEL: cir.func @foo
  // CIR: %[[C1:.*]] = cir.const #cir.int<1> : !s32i loc(#loc6)
  // CIR: %{{.*}} = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!ty_S>>, !cir.ptr<!ty_S>
  // CIR: cir.store %[[C1]], %{{.*}} : !s32i, !cir.ptr<!s32i> tbaa(#tbaa[[CHAR]])

  // LLVM-LABEL: void @foo
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa ![[TBAA_TAG:.*]]
  s->a = 1;
}

// LLVM: ![[TBAA_TAG]] = !{![[CHAR:.*]], ![[CHAR]], i64 0}
// LLVM: ![[CHAR]] = !{!"omnipotent char", ![[ROOT:.*]], i64 0}
// LLVM: ![[ROOT]] = !{!"Simple C/C++ TBAA"}
