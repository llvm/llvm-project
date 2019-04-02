// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

extern "C" void followup_inner(int n, int *x) {
#pragma unroll_and_jam(4)
  for(int j = 0; j < n; j++) {
#pragma clang loop pipeline_initiation_interval(10)
#pragma clang loop unroll_count(4)
#pragma clang loop vectorize(assume_safety)
#pragma clang loop distribute(enable)
    for(int i = 0; i < n; i++)
      x[j+i*n] = 10;

  }
}


// CHECK-LABEL: define void @followup_inner
// CHECK: br label %for.cond1, !llvm.loop ![[INNERLOOP_3:[0-9]+]]
// CHECK: br label %for.cond, !llvm.loop ![[OUTERLOOP_9:[0-9]+]]

// CHECK-DAG: ![[ACCESSGROUP_2:[0-9]+]] = distinct !{}

// CHECK-DAG: ![[INNERLOOP_3:[0-9]+]] = distinct !{![[INNERLOOP_3:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[DISTRIBUTE_5:[0-9]+]], ![[DISTRIBUTE_FOLLOWUP_6:[0-9]+]]}
// CHECK-DAG: ![[PARALLEL_ACCESSES_4:[0-9]+]] = !{!"llvm.loop.parallel_accesses", !2}
// CHECK-DAG: ![[DISTRIBUTE_5:[0-9]+]] = !{!"llvm.loop.distribute.enable", i1 true}
// CHECK-DAG: ![[DISTRIBUTE_FOLLOWUP_6:[0-9]+]] = !{!"llvm.loop.distribute.followup_all", ![[LOOP_7:[0-9]+]]}

// CHECK-DAG: ![[LOOP_7:[0-9]+]] = distinct !{![[LOOP_7:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[VECTORIZE_8:[0-9]+]]}
// CHECK-DAG: ![[VECTORIZE_8:[0-9]+]] = !{!"llvm.loop.vectorize.enable", i1 true}

// CHECK-DAG: ![[OUTERLOOP_9:[0-9]+]] = distinct !{![[OUTERLOOP_9:[0-9]+]], ![[UNROLLANDJAM_COUNT_10:[0-9]+]], ![[UNROLLANDJAM_FOLLOWUPINNER_11:[0-9]+]]}
// CHECK-DAG: ![[UNROLLANDJAM_COUNT_10:[0-9]+]] = !{!"llvm.loop.unroll_and_jam.count", i32 4}
// CHECK-DAG: ![[UNROLLANDJAM_FOLLOWUPINNER_11:[0-9]+]] = !{!"llvm.loop.unroll_and_jam.followup_inner", !12}

// CHECK-DAG: ![[LOOP_12:[0-9]+]] = distinct !{![[LOOP_12:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[ISVECTORIZED_13:[0-9]+]], ![[UNROLL_COUNT_13:[0-9]+]], ![[UNROLL_FOLLOWUP_14:[0-9]+]]}
// CHECK-DAG: ![[ISVECTORIZED_13:[0-9]+]] = !{!"llvm.loop.isvectorized"}
// CHECK-DAG: ![[UNROLL_COUNT_13:[0-9]+]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK-DAG: ![[UNROLL_FOLLOWUP_14:[0-9]+]] = !{!"llvm.loop.unroll.followup_all", ![[LOOP_15:[0-9]+]]}

// CHECK-DAG: ![[LOOP_15:[0-9]+]] = distinct !{![[LOOP_15:[0-9]+]], ![[PARALLEL_ACCESSES_4:[0-9]+]], ![[ISVECTORIZED_13:[0-9]+]], ![[UNROLL_DISABLE_16:[0-9]+]], ![[PIPELINE_17:[0-9]+]]}
// CHECK-DAG: ![[UNROLL_DISABLE_16:[0-9]+]] = !{!"llvm.loop.unroll.disable"}
// CHECK-DAG: ![[PIPELINE_17:[0-9]+]] = !{!"llvm.loop.pipeline.initiationinterval", i32 10}
