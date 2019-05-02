// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

extern "C" void followup_outer(int n, int *x) {
#pragma clang loop pipeline_initiation_interval(10)
#pragma clang loop unroll_count(4)
#pragma unroll_and_jam
#pragma clang loop vectorize(assume_safety)
#pragma clang loop distribute(enable)
  for(int j = 0; j < n; j++) {
    x[j] = 10;
  }
}


// CHECK-LABEL: define void @followup_outer
// CHECK: br label %for.cond, !llvm.loop ![[LOOP_3:[0-9]+]]

// CHECK-DAG: ![[ACCESSGROUP_2:[0-9]+]] = distinct !{}

// CHECK-DAG: ![[LOOP_3:[0-9]+]] = distinct !{![[LOOP_3:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[DISTRIBUTE_5:[0-9]+]], ![[DISTRIBUTE_FOLLOWUP_6:[0-9]+]]}
// CHECK-DAG: ![[PARALLEL_ACCESSES_4:[0-9]+]] = !{!"llvm.loop.parallel_accesses", ![[ACCESSGROUP_2]]}
// CHECK-DAG: ![[DISTRIBUTE_5:[0-9]+]] = !{!"llvm.loop.distribute.enable", i1 true}
// CHECK-DAG: ![[DISTRIBUTE_FOLLOWUP_6:[0-9]+]] = !{!"llvm.loop.distribute.followup_all", ![[LOOP_7:[0-9]+]]}

// CHECK-DAG: ![[LOOP_7:[0-9]+]] = distinct !{![[LOOP_7:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[VECTORIZE_8:[0-9]+]], ![[VECTORIZE_FOLLOWUP_9:[0-9]+]]}
// CHECK-DAG: ![[VECTORIZE_8:[0-9]+]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK-DAG: ![[VECTORIZE_FOLLOWUP_9:[0-9]+]] = !{!"llvm.loop.vectorize.followup_all", ![[LOOP_10:[0-9]+]]}

// CHECK-DAG: ![[LOOP_10:[0-9]+]] = distinct !{![[LOOP_10:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[ISVECTORIZED_11:[0-9]+]], ![[UNROLLANDJAM_12:[0-9]+]], ![[UNROLLANDJAM_FOLLOWUPOUTER_13:[0-9]+]]}
// CHECK-DAG: ![[ISVECTORIZED_11:[0-9]+]] = !{!"llvm.loop.isvectorized"}
// CHECK-DAG: ![[UNROLLANDJAM_12:[0-9]+]] =  !{!"llvm.loop.unroll_and_jam.enable"}
// CHECK-DAG: ![[UNROLLANDJAM_FOLLOWUPOUTER_13:[0-9]+]] = !{!"llvm.loop.unroll_and_jam.followup_outer", ![[LOOP_14:[0-9]+]]}

// CHECK-DAG: ![[LOOP_14:[0-9]+]] = distinct !{![[LOOP_14:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[ISVECTORIZED_11:[0-9]+]], ![[UNROLLANDJAM_DISABLE_15:[0-9]+]], ![[UNROLL_COUNT_16:[0-9]+]], ![[UNROLL_FOLLOWUP_17:[0-9]+]]}
// CHECK-DAG: ![[UNROLLANDJAM_DISABLE_15:[0-9]+]]  = !{!"llvm.loop.unroll_and_jam.disable"}
// CHECK-DAG: ![[UNROLL_COUNT_16:[0-9]+]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK-DAG: ![[UNROLL_FOLLOWUP_17:[0-9]+]] = !{!"llvm.loop.unroll.followup_all", ![[LOOP_18:[0-9]+]]}

// CHECK-DAG: ![[LOOP_18:[0-9]+]] = distinct !{![[LOOP_18:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[ISVECTORIZED_11:[0-9]+]], ![[UNROLLANDJAM_DISABLE_15:[0-9]+]], ![[UNROLL_DISABLE_19:[0-9]+]], ![[INITIATIONINTERVAL_20:[0-9]+]]}
// CHECK-DAG: ![[UNROLL_DISABLE_19:[0-9]+]]  = !{!"llvm.loop.unroll.disable"}
// CHECK-DAG: ![[INITIATIONINTERVAL_20:[0-9]+]] = !{!"llvm.loop.pipeline.initiationinterval", i32 10}
