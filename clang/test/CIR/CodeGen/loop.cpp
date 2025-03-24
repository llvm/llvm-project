// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void l0() {
  for (;;) {
  }
}

// CIR: cir.func @l0
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.condition(%[[TRUE]])
// CIR:     } body {
// CIR:       cir.yield
// CIR:     } step {
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define void @l0()
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l0v()
// OGCG: entry:
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_COND]]
