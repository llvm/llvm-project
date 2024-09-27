// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s -o - | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @foo(%arg0: !fir.ref<!fir.char<1,?>> {fir.bindc_name = "str1"} , %arg1: !fir.ref<i64> {fir.bindc_name = "len1"} loc("/home/haqadeer/work/fortran/t1/../str.f90":1:1), %arg2: i64) {
    %0 = fir.emboxchar %arg0, %arg2 : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
    %c4_i32 = arith.constant 4 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0_i64 = arith.constant 0 : i64
    %1 = fir.undefined !fir.dscope
    %2 = fircg.ext_declare %arg1 dummy_scope %1 {uniq_name = "_QFfooElen1"} : (!fir.ref<i64>, !fir.dscope) -> !fir.ref<i64> loc(#loc1)
    %3:2 = fir.unboxchar %0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %4 = fir.load %2 : !fir.ref<i64>
    %5 = arith.cmpi sgt, %4, %c0_i64 : i64
    %6 = arith.select %5, %4, %c0_i64 : i64
    %7 = fircg.ext_declare %3#0 typeparams %6 dummy_scope %1 {uniq_name = "_QFfooEstr1"} : (!fir.ref<!fir.char<1,?>>, i64, !fir.dscope) -> !fir.ref<!fir.char<1,?>> loc(#loc2)
    return
  } loc(#loc3)
}


#loc1 = loc("test.f90":18:1)
#loc2 = loc("test.f90":17:1)
#loc3 = loc("test.f90":15:1)

// CHECK: #[[VAR:.*]] = #llvm.di_local_variable<{{.*}}name = "._QFfooEstr1"{{.*}}flags = Artificial>
// CHECK: func.func @foo
// CHECK: llvm.intr.dbg.value #[[VAR]]
// CHECK: return
// CHECK: #[[STR_TY:.*]] = #llvm.di_string_type<tag = DW_TAG_string_type, name = "", stringLength = #[[VAR]], encoding = DW_ATE_ASCII>
// CHECK: #llvm.di_local_variable<{{.*}}name = "str1"{{.*}}type = #[[STR_TY]]>

