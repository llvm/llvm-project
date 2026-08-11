// RUN: inter-opt --split-input-file --inter-decompose-wide -verify-diagnostics %s

module {
  func.func @dynamic_multiply(%base: !llvm.ptr<1>, %lhs: i64, %rhs: i64) {
    // expected-error@+1 {{dynamic pointer-offset multiplication is not supported}}
    %offset = llvm.mul %lhs, %rhs : i64
    %ptr = xw.ptradd %base, %offset : !llvm.ptr<1>, i64
    return
  }
}

// -----

module {
  // expected-error@+1 {{packed pointer offset must be i32}}
  func.func @narrow_offset(%base: !llvm.ptr<1>, %offset: i16) {
    %ptr = xw.ptradd %base, %offset : !llvm.ptr<1>, i16
    return
  }
}
