// RUN: inter-opt --split-input-file --inter-normalize-cf -verify-diagnostics %s

module {
  // expected-error@+1 {{defined helper functions are not supported; inline calls before compiling with Inter}}
  llvm.func spir_funccc @helper() {
    llvm.return
  }
}

// -----

module {
  // expected-error@+1 {{variadic kernels are not supported}}
  llvm.func spir_kernelcc @variadic(...) {
    llvm.return
  }
}

// -----

module {
  // expected-error@+1 {{kernel return values are not supported}}
  llvm.func spir_kernelcc @returns_value() -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
}
