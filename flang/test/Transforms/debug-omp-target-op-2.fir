// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @fn_(%arg0: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "c"}, %arg2: !fir.ref<i32> {fir.bindc_name = "d"}) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = fir.alloca i32
    %1 = fir.alloca i32
    %2 = fir.undefined !fir.dscope
    %3 = fircg.ext_declare %arg1 dummy_scope %2 {uniq_name = "_QFfnEc"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc2)
    %4 = fircg.ext_declare %arg2 dummy_scope %2 {uniq_name = "_QFfnEd"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %5 = fir.load %3 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> index
    %9 = fir.load %4 : !fir.ref<i32>
    %10 = fir.convert %9 : (i32) -> index
    %15 = fircg.ext_declare %arg0(%6, %10) dummy_scope %2 {uniq_name = "_QFfnEb"} : (!fir.ref<!fir.array<?x?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?xi32>> loc(#loc4)
    %16 = fircg.ext_embox %15(%6, %10) : (!fir.ref<!fir.array<?x?xi32>>, index, index) -> !fir.box<!fir.array<?x?xi32>>
    %17:3 = fir.box_dims %16, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
    %18 = arith.subi %17#1, %c1 : index
    %19 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%18 : index) extent(%17#1 : index) stride(%17#2 : index) start_idx(%c1 : index) {stride_in_bytes = true}
    %20 = arith.muli %17#2, %17#1 : index
    %21:3 = fir.box_dims %16, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
    %22 = arith.subi %21#1, %c1 : index
    %23 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%22 : index) extent(%21#1 : index) stride(%20 : index) start_idx(%c1 : index) {stride_in_bytes = true}
    %24 = omp.map.info var_ptr(%15 : !fir.ref<!fir.array<?x?xi32>>, i32) map_clauses(tofrom) capture(ByRef) bounds(%19, %23) -> !fir.ref<!fir.array<?x?xi32>> {name = "b"}
    %25 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    %26 = omp.map.info var_ptr(%0 : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    omp.target map_entries(%24 -> %arg3, %25 -> %arg4, %26 -> %arg5 : !fir.ref<!fir.array<?x?xi32>>, !fir.ref<i32>, !fir.ref<i32>) {
      %27 = fir.load %arg5 : !fir.ref<i32>
      %28 = fir.load %arg4 : !fir.ref<i32>
      %29 = fir.convert %27 : (i32) -> index
      %31 = fir.convert %28 : (i32) -> index
      %37 = fircg.ext_declare %arg3(%29, %31) {uniq_name = "_QFfnEb"} : (!fir.ref<!fir.array<?x?xi32>>, index, index) -> !fir.ref<!fir.array<?x?xi32>> loc(#loc5)
      omp.terminator
    } loc(#loc6)
    return
  } loc(#loc7)
}
#loc1 = loc("test.f90":1:1)
#loc2 = loc("test.f90":3:1)
#loc3 = loc("test.f90":7:1)
#loc4 = loc("test.f90":8:1)
#loc5 = loc("test.f90":6:1)
#loc6 = loc("test.f90":16:1)
#loc7 = loc("test.f90":26:1)


// Test that variable size arrays inside target regions get their own
// compiler generated variables for size.

// CHECK: #[[SP:.*]] = #llvm.di_subprogram<{{.*}}name = "__omp_offloading_{{.*}}_fn__l16"{{.*}}>
// CHECK: #llvm.di_local_variable<scope = #[[SP]], name = "._QFfnEb1"{{.*}}>
// CHECK: #llvm.di_local_variable<scope = #[[SP]], name = "._QFfnEb2"{{.*}}>
