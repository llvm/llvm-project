// RUN: inter-opt --split-input-file --inter-select-to-machine -verify-diagnostics %s
// CHECK-NOT: llvm

module {
  func.func @unsupported_xw() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %one = xw.constant 1 : i32
    %lanes = xw.splat %one : i32 -> !xw.simd<i32, 8>
    // expected-error@+1 {{integer operation has no XeMachine instruction selection}}
    %division = xw.binary divui %lanes, %lanes : !xw.simd<i32, 8>, !xw.simd<i32, 8> -> !xw.simd<i32, 8>
    return
  }
}

// -----

module {
  func.func @unsupported_poison() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    // expected-error@+1 {{unsupported UB poison result type 'vector<2xi32>'}}
    %poison = ub.poison : vector<2xi32>
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
    // expected-error@+1 {{selector accepts only func, scf, and XW operations}}
    %zero = arith.constant 0 : i32
    %one = arith.addi %zero, %zero : i32
    return
  }
}
