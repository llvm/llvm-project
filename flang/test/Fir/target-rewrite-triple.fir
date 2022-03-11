// RUN: fir-opt --target-rewrite %s | FileCheck %s --check-prefix=UNCHANGED
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=CHANGED

// UNCHANGED: llvm.target_triple = "aarch64-unknown-linux-gnu"
// CHANGED: llvm.target_triple = "x86_64-unknown-linux-gnu"
// CHANGED-NOT: llvm.target_triple = "aarch64-unknown-linux-gnu"
module attributes {llvm.target_triple = "aarch64-unknown-linux-gnu"}  {
  func @dummyfunc() -> () {
    return
  }
}

