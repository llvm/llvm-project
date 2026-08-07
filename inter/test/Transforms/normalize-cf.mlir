// inter-normalize-cf: llvm.func -> func.func, llvm branches -> cf branches.
// RUN: inter-opt %s --inter-normalize-cf | FileCheck %s

module {
  // CHECK-NOT: llvm.func @k
  // CHECK: func.func @k(%{{.*}}: !llvm.ptr<1>) attributes {xemachine.kernel}
  llvm.func spir_kernelcc @k(%arg0: !llvm.ptr<1>) {
    %c = llvm.mlir.constant(true) : i1
    // CHECK: cf.cond_br
    // CHECK-NOT: llvm.cond_br
    llvm.cond_br %c, ^bb1, ^bb2
  ^bb1:
    // CHECK: cf.br
    // CHECK-NOT: llvm.br
    llvm.br ^bb2
  ^bb2:
    // CHECK: return
    // CHECK-NOT: llvm.return
    llvm.return
  }
  // The declaration must survive untouched.
  // CHECK: llvm.func {{.*}}@declared(
  llvm.func spir_funccc @declared(i32) -> i64
}
