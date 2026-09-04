// RUN: inter-opt --split-input-file --inter-select-to-machine -verify-diagnostics %s
// CHECK-NOT: llvm

module {
  func.func @unsupported_poison() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    // expected-error@+1 {{unsupported UB poison result type 'vector<[2]xi32>'}}
    %poison = ub.poison : vector<[2]xi32>
    return
  }
}

// -----

module {
  func.func @unsupported_ub() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    // expected-error@+1 {{selector accepts only fully poisoned ub.poison operations}}
    ub.unreachable
  }
}

// -----

module {
  func.func @unsupported_dialect() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %zero = arith.constant 0 : i32
    // expected-error@+1 {{selector accepts only func, scf, selected arith, and XW operations}}
    %one = arith.addi %zero, %zero : i32
    return
  }
}
