// RUN: inter-opt --split-input-file --inter-import-llvm -verify-diagnostics %s

// expected-error@+1 {{unsupported LLVM target triple 'x86_64-unknown-linux-gnu'}}
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
}

// -----

// expected-error@+1 {{LLVM module assembly is unsupported}}
module attributes {llvm.module_asm = [".byte 0"]} {
}

// -----

// expected-error@+1 {{LLVM pointer layout for address space 0 must be 64 bits with 8-byte ABI alignment}}
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    !llvm.ptr = dense<32> : vector<4xi64>,
    "dlti.endianness" = "little"
  >
} {
}

// -----

module {
  // expected-error@+1 {{defined helpers must be inlined before LLVM import}}
  llvm.func @helper() {
    llvm.return
  }
}

// -----

module {
  // expected-error@+1 {{variadic kernels are unsupported}}
  llvm.func spir_kernelcc @variadic(...) {
    llvm.return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument 0 has unsupported scalar width 16}}
  llvm.func spir_kernelcc @half(%arg: f16) {
    llvm.return
  }
}
