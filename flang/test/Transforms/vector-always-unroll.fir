// RUN: fir-opt --vector-always-unroll %s | FileCheck %s
// RUN: fir-opt --vector-always-unroll="max-unroll-ops=1" %s | FileCheck %s --check-prefix=CAPPED
// RUN: fir-opt --vector-always-unroll="max-unroll-ops=50" %s | FileCheck %s --check-prefix=CAP50

// Vectorization directives (`!dir$ vector always`, `!dir$ vector length`,
// `!dir$ simd`) lower to a vectorize.enable (disable = false) loop_annotation.
// For each such loop this pass tags every nested fir.do_loop with
// llvm.loop.unroll.full (when trip counts are constant and within thresholds),
// leaving the outer loop untouched.

#vec_enable = #llvm.loop_vectorize<disable = false>
#anno_vec = #llvm.loop_annotation<vectorize = #vec_enable>
#vec_width = #llvm.loop_vectorize<disable = false, scalableEnable = false, width = 4 : i64>
#anno_width = #llvm.loop_annotation<vectorize = #vec_width>
#vec_disable = #llvm.loop_vectorize<disable = true>
#anno_novec = #llvm.loop_annotation<vectorize = #vec_disable>

// CHECK: #[[UNROLL:[a-zA-Z0-9_]+]] = #llvm.loop_unroll<full = true>
// CHECK: #[[UNROLL_ANNO:[a-zA-Z0-9_]+]] = #llvm.loop_annotation<unroll = #[[UNROLL]]>

// CHECK-LABEL: func.func @tag_middle
// CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #{{.+}}}
// CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[UNROLL_ANNO]]}
// CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[UNROLL_ANNO]]}

// CAPPED-LABEL: func.func @tag_middle
// CAPPED-NOT: unroll
func.func @tag_middle() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_loop %i = %c1 to %c10 step %c1 attributes {loopAnnotation = #anno_vec} {
    fir.do_loop %j = %c1 to %c10 step %c1 {
      fir.do_loop %k = %c1 to %c10 step %c1 {
        %0 = arith.addi %i, %j : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @tag_middle_width
// CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #{{.+}}}
// CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[UNROLL_ANNO]]}
// CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[UNROLL_ANNO]]}
func.func @tag_middle_width() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_loop %i = %c1 to %c10 step %c1 attributes {loopAnnotation = #anno_width} {
    fir.do_loop %j = %c1 to %c10 step %c1 {
      fir.do_loop %k = %c1 to %c10 step %c1 {
        %0 = arith.addi %i, %j : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @skip_too_many_ops
// CHECK-NOT: unroll
func.func @skip_too_many_ops() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  fir.do_loop %i = %c1 to %c10 step %c1 attributes {loopAnnotation = #anno_vec} {
    fir.do_loop %j = %c1 to %c100 step %c1 {
      fir.do_loop %k = %c1 to %c200 step %c1 {
        %0 = arith.addi %i, %j : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @skip_nonconst
// CHECK-NOT: unroll
func.func @skip_nonconst(%n: index) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_loop %i = %c1 to %c10 step %c1 attributes {loopAnnotation = #anno_vec} {
    fir.do_loop %j = %c1 to %n step %c1 {
      fir.do_loop %k = %c1 to %c10 step %c1 {
        %0 = arith.addi %i, %j : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @skip_novector
// CHECK-NOT: unroll
func.func @skip_novector() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_loop %i = %c1 to %c10 step %c1 attributes {loopAnnotation = #anno_novec} {
    fir.do_loop %j = %c1 to %c10 step %c1 {
      fir.do_loop %k = %c1 to %c10 step %c1 {
        %0 = arith.addi %i, %j : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @no_annotation
// CHECK-NOT: loopAnnotation
func.func @no_annotation() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_loop %i = %c1 to %c10 step %c1 {
    fir.do_loop %j = %c1 to %c10 step %c1 {
      fir.do_loop %k = %c1 to %c10 step %c1 {
        %0 = arith.addi %i, %j : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @omp_collapse_counted
// CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #{{.+}}}
// CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[UNROLL_ANNO]]}

// With max-unroll-ops=50: ignoring the collapsed loop the estimate would be 4
// (<= 50, so the loop would be tagged). Counting the 100-iteration collapsed
// omp.loop_nest raises it to 404 (> 50), so nothing is tagged.
// CAP50-LABEL: func.func @omp_collapse_counted
// CAP50-NOT: unroll
func.func @omp_collapse_counted() {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  fir.do_loop %i = %c1 to %c4 step %c1 attributes {loopAnnotation = #anno_vec} {
    fir.do_loop %j = %c1 to %c4 step %c1 {
      omp.simd {
        omp.loop_nest (%k, %l) : i32 = (%c1_i32, %c1_i32) to (%c10_i32, %c10_i32) inclusive step (%c1_i32, %c1_i32) collapse(2) {
          %0 = arith.addi %k, %l : i32
          omp.yield
        }
      }
    }
  }
  return
}
// CHECK-LABEL: func.func @omp_collapse_nonconst
// CHECK-NOT: unroll
func.func @omp_collapse_nonconst(%n: i32) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  fir.do_loop %i = %c1 to %c4 step %c1 attributes {loopAnnotation = #anno_vec} {
    fir.do_loop %j = %c1 to %c4 step %c1 {
      omp.simd {
        omp.loop_nest (%k, %l) : i32 = (%c1_i32, %c1_i32) to (%n, %n) inclusive step (%c1_i32, %c1_i32) collapse(2) {
          %0 = arith.addi %k, %l : i32
          omp.yield
        }
      }
    }
  }
  return
}
