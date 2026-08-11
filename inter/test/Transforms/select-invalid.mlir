// RUN: inter-opt --inter-select-to-machine -verify-diagnostics %s

module {
  func.func @unsupported() attributes {xemachine.kernel} {
    %one = llvm.mlir.constant(1 : i32) : i32
    // expected-error@+1 {{unsupported operation during Inter machine selection}}
    %product = llvm.mul %one, %one : i32
    return
  }
}
