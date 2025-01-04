// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Fortran source code:
//
// subroutine TestTargetData(p, a, b)
//   real    ::  p(10), a(10), b(10)
//   !$omp target data map(from: p)
//      !$omp target map(to: a )
//            p(1) = a(1)
//      !$omp end target
//      !$omp target map(to: b )
//            p(1) = b(1)
//      !$omp end target
//   !$omp end target data
// end subroutine

// CHECK-LABEL: Testing : "_QPTestTargetData"

// CHECK-DAG: targetArrayA#0 <-> targetArrayP#0: NoAlias
// CHECK-DAG: targetArrayA#0 <-> targetArrayB#0: NoAlias
// CHECK-DAG: targetArrayP#0 <-> targetArrayB#0: NoAlias

func.func @_QPTestTargetData(%arg0: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "p"}, %arg1: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "a"}, %arg2: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "b"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg1(%1) dummy_scope %0 {uniq_name = "_QFtest_target_dataEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %c10_0 = arith.constant 10 : index
  %3 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg2(%3) dummy_scope %0 {uniq_name = "_QFtest_target_dataEb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %c10_1 = arith.constant 10 : index
  %5 = fir.shape %c10_1 : (index) -> !fir.shape<1>
  %6:2 = hlfir.declare %arg0(%5) dummy_scope %0 {uniq_name = "_QFtest_target_dataEp"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %7 = arith.subi %c10_1, %c1 : index
  %8 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%7 : index) extent(%c10_1 : index) stride(%c1 : index) start_idx(%c1 : index)
  %9 = omp.map.info var_ptr(%6#1 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(from) capture(ByRef) bounds(%8) -> !fir.ref<!fir.array<10xf32>> {name = "p"}
  omp.target_data map_entries(%9 : !fir.ref<!fir.array<10xf32>>) {
    %c1_2 = arith.constant 1 : index
    %c0_3 = arith.constant 0 : index
    %10 = arith.subi %c10, %c1_2 : index
    %11 = omp.map.bounds lower_bound(%c0_3 : index) upper_bound(%10 : index) extent(%c10 : index) stride(%c1_2 : index) start_idx(%c1_2 : index)
    %12 = omp.map.info var_ptr(%2#1 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(to) capture(ByRef) bounds(%11) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
    %c1_4 = arith.constant 1 : index
    %c0_5 = arith.constant 0 : index
    %13 = arith.subi %c10_1, %c1_4 : index
    %14 = omp.map.bounds lower_bound(%c0_5 : index) upper_bound(%13 : index) extent(%c10_1 : index) stride(%c1_4 : index) start_idx(%c1_4 : index)
    %15 = omp.map.info var_ptr(%6#1 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%14) -> !fir.ref<!fir.array<10xf32>> {name = "p"}
    omp.target map_entries(%12 -> %arg3, %15 -> %arg4 : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
      %c10_10 = arith.constant 10 : index
      %22 = fir.shape %c10_10 : (index) -> !fir.shape<1>
      %23:2 = hlfir.declare %arg3(%22) {uniq_name = "_QFtest_target_dataEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
      %c10_11 = arith.constant 10 : index
      %24 = fir.shape %c10_11 : (index) -> !fir.shape<1>
      %25:2 = hlfir.declare %arg4(%24) {uniq_name = "_QFtest_target_dataEp"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
      %c1_12 = arith.constant 1 : index
      %26 = hlfir.designate %23#0 (%c1_12) {test.ptr = "targetArrayA"}  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      %27 = fir.load %26 : !fir.ref<f32>
      %c1_13 = arith.constant 1 : index
      %28 = hlfir.designate %25#0 (%c1_13) {test.ptr = "targetArrayP"}  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      hlfir.assign %27 to %28 : f32, !fir.ref<f32>
      omp.terminator
    }
    %c1_6 = arith.constant 1 : index
    %c0_7 = arith.constant 0 : index
    %16 = arith.subi %c10_0, %c1_6 : index
    %17 = omp.map.bounds lower_bound(%c0_7 : index) upper_bound(%16 : index) extent(%c10_0 : index) stride(%c1_6 : index) start_idx(%c1_6 : index)
    %18 = omp.map.info var_ptr(%4#1 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(to) capture(ByRef) bounds(%17) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
    %c1_8 = arith.constant 1 : index
    %c0_9 = arith.constant 0 : index
    %19 = arith.subi %c10_1, %c1_8 : index
    %20 = omp.map.bounds lower_bound(%c0_9 : index) upper_bound(%19 : index) extent(%c10_1 : index) stride(%c1_8 : index) start_idx(%c1_8 : index)
    %21 = omp.map.info var_ptr(%6#1 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%20) -> !fir.ref<!fir.array<10xf32>> {name = "p"}
    omp.target map_entries(%18 -> %arg3, %21 -> %arg4 : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
      %c10_10 = arith.constant 10 : index
      %22 = fir.shape %c10_10 : (index) -> !fir.shape<1>
      %23:2 = hlfir.declare %arg3(%22) {uniq_name = "_QFtest_target_dataEb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
      %c10_11 = arith.constant 10 : index
      %24 = fir.shape %c10_11 : (index) -> !fir.shape<1>
      %25:2 = hlfir.declare %arg4(%24) {uniq_name = "_QFtest_target_dataEp"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
      %c1_12 = arith.constant 1 : index
      %26 = hlfir.designate %23#0 (%c1_12) {test.ptr = "targetArrayB"}  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      %27 = fir.load %26 : !fir.ref<f32>
      %c1_13 = arith.constant 1 : index
      %28 = hlfir.designate %25#0 (%c1_13) {test.ptr = "targetArrayP"}  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      hlfir.assign %27 to %28 : f32, !fir.ref<f32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

