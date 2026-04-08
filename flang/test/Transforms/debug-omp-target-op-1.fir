// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s
// RUN: fir-opt --add-debug-info="debug-level=LineTablesOnly" --mlir-print-debuginfo %s | FileCheck %s --check-prefix=LINETABLE

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QQmain() attributes {fir.bindc_name = "test"} {
    %c13_i32 = arith.constant 13 : i32
    %c12_i32 = arith.constant 12 : i32
    %c6_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"} loc(#loc1)
    %1 = fircg.ext_declare %0 {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc1)
    %2 = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"} loc(#loc2)
    %3 = fircg.ext_declare %2 {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc2)
    %4 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "x"}
    %5 = omp.map.info var_ptr(%3 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "y"}
    omp.target map_entries(%4 -> %arg0, %5 -> %arg1 : !fir.ref<i32>, !fir.ref<i32>) {
      %16 = fircg.ext_declare %arg0 {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc3)
      %17 = fircg.ext_declare %arg1 {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc4)
      omp.terminator
    } loc(#loc5)
    return
  }
}
#loc1 = loc("test.f90":1:1)
#loc2 = loc("test.f90":3:1)
#loc3 = loc("test.f90":7:1)
#loc4 = loc("test.f90":8:1)
#loc5 = loc("test.f90":6:1)

// CHECK: #[[SP:.*]] = #llvm.di_subprogram<{{.*}}name = "test"{{.*}}>
// CHECK: #[[SP1:.*]] = #llvm.di_subprogram<{{.*}}name = "__omp_offloading_{{.*}}_QQmain_l6"{{.*}}line = 6{{.*}}subprogramFlags = "LocalToUnit|Definition"{{.*}}>
// CHECK: #llvm.di_local_variable<scope = #[[SP]], name = "x"{{.*}}line = 1, type = #[[TY:.*]]>
// CHECK: #llvm.di_local_variable<scope = #[[SP]], name = "y"{{.*}}line = 3, type = #[[TY]]>
// CHECK: #llvm.di_local_variable<scope = #[[SP1]], name = "x"{{.*}}line = 7, type = #[[TY]]>
// CHECK: #llvm.di_local_variable<scope = #[[SP1]], name = "y"{{.*}}line = 8, type = #[[TY]]>

// LINETABLE: #[[SP:.*]] = #llvm.di_subprogram<{{.*}}name = "test"{{.*}}>
// LINETABLE: #[[SP1:.*]] = #llvm.di_subprogram<{{.*}}name = "__omp_offloading_{{.*}}_QQmain_l6"{{.*}}line = 6{{.*}}subprogramFlags = "LocalToUnit|Definition"{{.*}}>
// LINETABLE-NOT: #llvm.di_local_variable
