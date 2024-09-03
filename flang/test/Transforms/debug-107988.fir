// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s -o - | FileCheck %s
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s -o - | \
// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" \
// RUN: --mlir-print-debuginfo  | FileCheck --check-prefix=CODEGEN %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @test(%arg0: !fir.ref<!fir.char<1,?>> {fir.bindc_name = "str"}, %arg1: i64) {
    %0 = fir.emboxchar %arg0, %arg1 : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
    %1 = fir.undefined !fir.dscope
    %2:2 = fir.unboxchar %0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)  loc(#loc1)
    %3 = fircg.ext_declare %2#0 typeparams %2#1 dummy_scope %1 {uniq_name = "_QFtestEstr"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,?>> loc(#loc1)
    return
  } loc(#loc2)
}

#loc1 = loc("test.f90":5:1)
#loc2 = loc("test.f90":15:1)


// CHECK: fir.unboxchar{{.*}} loc(#[[LOC1:.*]])
// CHECK: #[[LOC2:.*]] = loc("test.f90":5:1)
// CHECK: #[[VAR:.*]] = #llvm.di_local_variable<{{.*}}name = "._QFtestEstr"{{.*}}flags = Artificial>
// CHECK: #[[STR_TY:.*]] = #llvm.di_string_type<tag = DW_TAG_string_type, name = "", stringLength = #[[VAR]], encoding = DW_ATE_ASCII>
// CHECK: #[[LOC1]] = loc(fused<#di_local_variable>[#[[LOC2]]])
// CHECK: #llvm.di_local_variable<{{.*}}name = "str"{{.*}}type = #[[STR_TY]]>

// CODEGEN: #[[VAR1:.*]] = #llvm.di_local_variable<{{.*}}name = "._QFtestEstr"{{.*}}flags = Artificial>
// CODEGEN: llvm.func @test
// CODEGEN: llvm.intr.dbg.value #[[VAR1]]
