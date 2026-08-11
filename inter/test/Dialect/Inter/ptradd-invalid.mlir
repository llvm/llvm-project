// RUN: inter-opt --split-input-file -verify-diagnostics %s

module {
  func.func @inbounds_without_nusw(%base: !llvm.ptr<1>, %offset: i64) {
    // expected-error@+1 {{inbounds must imply nusw}}
    %ptr = xw.ptradd %base, %offset {gep_flags = 1 : i32}
        : !llvm.ptr<1>, i64
    return
  }
}

// -----

module {
  func.func @unknown_flags(%base: !llvm.ptr<1>, %offset: i64) {
    // expected-error@+1 {{has unknown LLVM GEP no-wrap flag bits}}
    %ptr = xw.ptradd %base, %offset {gep_flags = 8 : i32}
        : !llvm.ptr<1>, i64
    return
  }
}
