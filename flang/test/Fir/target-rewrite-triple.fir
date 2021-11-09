// RUN: fir-opt --target-rewrite %s | FileCheck %s --check-prefix=UNCHANGED
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=CHANGED

// UNCHANGED: fir.triple = "aarch64-unknown-linux-gnu"
// CHANGED: fir.triple = "x86_64-unknown-linux-gnu"
// CHANGED-NOT: fir.triple = "aarch64-unknown-linux-gnu"
module attributes {fir.triple = "aarch64-unknown-linux-gnu"}  {
  func @dummyfunc() -> () {
    return
  }
}

