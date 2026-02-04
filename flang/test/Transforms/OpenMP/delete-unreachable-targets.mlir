// RUN: fir-opt --omp-delete-unreachable-targets %s | FileCheck %s

// This test verifies that OpenMP target operations in unreachable code are
// deleted.


// CHECK-LABEL: func.func @test_if_false_simple
func.func @test_if_false_simple() {
  %false = arith.constant false
  // The target in the dead branch should be removed
  // CHECK: fir.if %false {
  // CHECK-NOT: omp.target
  // CHECK: }
  fir.if %false {
    omp.target {
      omp.terminator
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_if_true_simple
func.func @test_if_true_simple() {
  %true = arith.constant true
  // The target should remain since the branch is reachable
  // CHECK: omp.target
  fir.if %true {
    omp.target {
      omp.terminator
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_nested_outer_false
func.func @test_nested_outer_false() {
  %false = arith.constant false
  %true = arith.constant true
  // Outer false makes the whole nested structure unreachable
  // CHECK: fir.if %false {
  // CHECK-NOT: omp.target
  // CHECK: }
  fir.if %false {
    fir.if %true {
      omp.target {
        omp.terminator
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_nested_inner_false
func.func @test_nested_inner_false() {
  %false = arith.constant false
  %true = arith.constant true
  // Outer true, inner false - target should be removed
  // CHECK: fir.if %true {
  // CHECK: fir.if %false {
  // CHECK-NOT: omp.target
  // CHECK: }
  fir.if %true {
    fir.if %false {
      omp.target {
        omp.terminator
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_nested_both_true
func.func @test_nested_both_true() {
  %true1 = arith.constant true
  %true2 = arith.constant true
  // CHECK: omp.target
  fir.if %true1 {
    fir.if %true2 {
      omp.target {
        omp.terminator
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_mixed_targets
func.func @test_mixed_targets() {
  %false = arith.constant false
  %true = arith.constant true

  // Live target - should remain (expect 2 targets total in output)
  // CHECK: omp.target
  omp.target {
    omp.terminator
  }

  // Another live target in if (true) - should remain
  // CHECK: omp.target
  fir.if %true {
    omp.target {
      omp.terminator
    }
  }

  // Dead target - will be removed
  // CHECK-NOT: omp.target
  fir.if %false {
    omp.target {
      omp.terminator
    }
  }

  return
}

// -----

// CHECK-LABEL: func.func @test_multiple_dead_targets
func.func @test_multiple_dead_targets() {
  %false = arith.constant false

  // All targets inside dead branch should be removed
  // CHECK-NOT: omp.target
  fir.if %false {
    omp.target {
      omp.terminator
    }

    omp.target {
      omp.terminator
    }

    omp.target {
      omp.terminator
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_if_else_false
func.func @test_if_else_false() {
  %false = arith.constant false

  // CHECK: fir.if %false {
  fir.if %false {
    // Then branch is unreachable, target should be deleted
    omp.target {
      omp.terminator
    }
  } else {
    // CHECK-NOT: omp.target
    // CHECK: } else {
    // Else branch is reachable, target should remain
    // CHECK: omp.target
    omp.target {
      omp.terminator
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_runtime_condition
func.func @test_runtime_condition(%arg0: i1) {
  // Runtime condition - cannot be optimized, should remain unchanged
  // CHECK: fir.if %arg0 {
  fir.if %arg0 {
    // CHECK: omp.target
    omp.target {
      omp.terminator
    }
  }
  return
}

// -----

// Test that targets nested in structured control flow within unreachable blocks
// are correctly identified as unreachable
// CHECK-LABEL: func.func @test_nested_in_unreachable_block
func.func @test_nested_in_unreachable_block() {
  cf.br ^bb2
^bb1:
  // This entire block is unreachable
  // Even though the fir.if condition is true, the whole block is dead
  %true = arith.constant true
  // CHECK: ^bb1:
  // CHECK-NOT: omp.target
  // CHECK: cf.br ^bb2
  fir.if %true {
    omp.target {
      omp.terminator
    }
  }
  cf.br ^bb2
^bb2:
  // CHECK: ^bb2:
  // CHECK-NEXT: omp.target
  omp.target {
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_unreachable_block_after_branch
func.func @test_unreachable_block_after_branch() {
  cf.br ^bb2
^bb1:
  // This block is unreachable - no predecessor branches to it
  // CHECK: ^bb1:
  // CHECK-NOT: omp.target
  // CHECK: cf.br ^bb2
  omp.target {
    omp.terminator
  }
  cf.br ^bb2
^bb2:
  // This block is reachable
  // CHECK: ^bb2:
  // CHECK-NEXT: omp.target
  omp.target {
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_multiple_unreachable_blocks
func.func @test_multiple_unreachable_blocks() {
  cf.br ^bb3
^bb1:
  // Unreachable block - no predecessor branches to it
  // CHECK: ^bb1:
  // CHECK-NOT: omp.target
  // CHECK: cf.br ^bb2
  omp.target {
    omp.terminator
  }
  cf.br ^bb2
^bb2:
  // Also unreachable - only reachable from ^bb1 which is itself unreachable
  // CHECK: ^bb2:
  // CHECK-NOT: omp.target
  // CHECK: return
  omp.target {
    omp.terminator
  }
  return
^bb3:
  // Reachable from entry
  // CHECK: ^bb3:
  // CHECK-NEXT: omp.target
  omp.target {
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_both_branches_reachable
func.func @test_both_branches_reachable(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK: ^bb1:
  // CHECK-NEXT: omp.target
  omp.target {
    omp.terminator
  }
  cf.br ^bb3
^bb2:
  // CHECK: ^bb2:
  // CHECK-NEXT: omp.target
  omp.target {
    omp.terminator
  }
  cf.br ^bb3
^bb3:
  return
}

// -----

// CHECK-LABEL: func.func @test_disconnected_block
func.func @test_disconnected_block() {
  // Entry goes directly to exit
  cf.br ^bb2
^bb1:
  // This block is completely disconnected - no way to reach it
  // CHECK: ^bb1:
  // CHECK-NOT: omp.target
  // CHECK: cf.br ^bb2
  omp.target {
    omp.terminator
  }
  cf.br ^bb2
^bb2:
  // Reachable from entry
  // CHECK: ^bb2:
  // CHECK-NEXT: omp.target
  omp.target {
    omp.terminator
  }
  return
}
