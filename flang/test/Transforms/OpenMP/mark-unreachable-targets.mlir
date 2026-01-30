// RUN: fir-opt --omp-mark-unreachable-targets %s | FileCheck %s

// CHECK-LABEL: func.func @test_if_false_simple
func.func @test_if_false_simple() {
  %false = arith.constant false
  // CHECK: fir.if %{{.*}} {
  fir.if %false {
    // CHECK: omp.target
    // CHECK-NEXT: omp.terminator
    // CHECK-NEXT: } {omp.target_unreachable}
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
  // CHECK: fir.if %{{.*}} {
  fir.if %true {
    // CHECK: omp.target {
    // CHECK-NEXT: omp.terminator
    // CHECK-NEXT: }
    // CHECK-NOT: omp.target_unreachable
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
  // CHECK: fir.if %{{.*}} {
  fir.if %false {
    // CHECK: fir.if %{{.*}} {
    fir.if %true {
      // CHECK: omp.target
      // CHECK-NEXT: omp.terminator
      // CHECK-NEXT: } {omp.target_unreachable}
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
  // CHECK: fir.if %{{.*}} {
  fir.if %true {
    // CHECK: fir.if %{{.*}} {
    fir.if %false {
      // CHECK: omp.target
      // CHECK: } {omp.target_unreachable}
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
  // CHECK: fir.if %{{.*}} {
  fir.if %true1 {
    // CHECK: fir.if %{{.*}} {
    fir.if %true2 {
      // CHECK: omp.target {
      // CHECK-NEXT: omp.terminator
      // CHECK-NEXT: }
      // CHECK-NOT: omp.target_unreachable
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

  // Dead target
  // CHECK: fir.if %{{.*}} {
  fir.if %false {
    // CHECK: omp.target
    // CHECK: } {omp.target_unreachable}
    omp.target {
      omp.terminator
    }
  }

  // Live target - should NOT have unreachable attribute
  // CHECK: omp.target {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  // CHECK-NOT: omp.target_unreachable
  omp.target {
    omp.terminator
  }

  // Another live target in if (true)
  // CHECK: fir.if %{{.*}} {
  fir.if %true {
    // CHECK: omp.target {
    // CHECK-NEXT: omp.terminator
    // CHECK-NEXT: }
    // CHECK-NOT: omp.target_unreachable
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

  // CHECK: fir.if %{{.*}} {
  fir.if %false {
    // CHECK: omp.target
    // CHECK: } {omp.target_unreachable}
    omp.target {
      omp.terminator
    }

    // CHECK: omp.target
    // CHECK: } {omp.target_unreachable}
    omp.target {
      omp.terminator
    }

    // CHECK: omp.target
    // CHECK: } {omp.target_unreachable}
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

  // CHECK: fir.if %{{.*}} {
  fir.if %false {
    // CHECK: omp.target
    // CHECK: } {omp.target_unreachable}
    omp.target {
      omp.terminator
    }
  } else {
    // Else branch should not be marked (it's reachable)
    // CHECK: omp.target {
    // CHECK-NEXT: omp.terminator
    // CHECK-NEXT: }
    // CHECK-NOT: omp.target_unreachable
    omp.target {
      omp.terminator
    }
  }
  return
}

// -----

// Test with cf.cond_br
// CHECK-LABEL: func.func @test_cf_cond_br_false
func.func @test_cf_cond_br_false() {
  %false = arith.constant false
  // CHECK: cf.cond_br %{{.*}}, ^bb1, ^bb2
  cf.cond_br %false, ^bb1, ^bb2
^bb1:
  // CHECK: omp.target
  // CHECK: } {omp.target_unreachable}
  omp.target {
    omp.terminator
  }
  cf.br ^bb2
^bb2:
  return
}

// -----

// CHECK-LABEL: func.func @test_cf_cond_br_true
func.func @test_cf_cond_br_true() {
  %true = arith.constant true
  // CHECK: cf.cond_br %{{.*}}, ^bb1, ^bb2
  cf.cond_br %true, ^bb1, ^bb2
^bb1:
  // CHECK: omp.target {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  // CHECK-NOT: omp.target_unreachable
  omp.target {
    omp.terminator
  }
  cf.br ^bb2
^bb2:
  return
}

// -----

// CHECK-LABEL: func.func @test_runtime_condition
func.func @test_runtime_condition(%arg0: i1) {
  // CHECK: fir.if %arg0 {
  fir.if %arg0 {
    // Runtime condition - should NOT be marked
    // CHECK: omp.target {
    // CHECK-NEXT: omp.terminator
    // CHECK-NEXT: }
    // CHECK-NOT: omp.target_unreachable
    omp.target {
      omp.terminator
    }
  }
  return
}

// -----

// Test for multiple predecessors - one reachable, one unreachable
// The block should NOT be marked as unreachable if ANY path is reachable
// CHECK-LABEL: func.func @test_multiple_predecessors
func.func @test_multiple_predecessors() {
  %false = arith.constant false
  cf.cond_br %false, ^bb2, ^bb1
^bb1:
  // Reachable path to bb2
  cf.br ^bb2
^bb2:
  // This block has two predecessors:
  // - bb0 with false condition (unreachable path)
  // - bb1 with unconditional branch (reachable path)
  // Target should NOT be marked unreachable because bb1 provides a reachable path
  // CHECK: omp.target {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  // CHECK-NOT: omp.target_unreachable
  omp.target {
    omp.terminator
  }
  return
}

// -----

// Test for multiple predecessors - ALL unreachable
// CHECK-LABEL: func.func @test_multiple_predecessors_all_unreachable
func.func @test_multiple_predecessors_all_unreachable() {
  %false1 = arith.constant false
  %false2 = arith.constant false
  cf.cond_br %false1, ^bb3, ^bb1
^bb1:
  cf.cond_br %false2, ^bb3, ^bb2
^bb2:
  cf.br ^bb4
^bb3:
  // This block has two predecessors:
  // - bb0 with false condition to bb3 (unreachable)
  // - bb1 with false condition to bb3 (unreachable)
  // Target SHOULD be marked unreachable because ALL paths are unreachable
  // CHECK: omp.target
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: } {omp.target_unreachable}
  omp.target {
    omp.terminator
  }
  cf.br ^bb4
^bb4:
  return
}

// -----

// Test for multiple predecessors with mixed constant and runtime conditions
// CHECK-LABEL: func.func @test_multiple_predecessors_mixed
func.func @test_multiple_predecessors_mixed(%arg0: i1) {
  %false = arith.constant false
  cf.cond_br %false, ^bb2, ^bb1
^bb1:
  // Runtime condition - could branch to bb2
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:
  // This block has two predecessors:
  // - bb0 with false condition (unreachable)
  // - bb1 with runtime condition (potentially reachable)
  // Target should NOT be marked because we can't prove bb1 path is unreachable
  // CHECK: omp.target {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
  // CHECK-NOT: omp.target_unreachable
  omp.target {
    omp.terminator
  }
  cf.br ^bb3
^bb3:
  return
}
