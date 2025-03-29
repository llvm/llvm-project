// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QQmain() attributes {fir.bindc_name = "mn"} {
    %0 = fir.alloca i32 {bindc_name = "i4", uniq_name = "_QFEi4"}
    %1 = fircg.ext_declare %0 {uniq_name = "_QFEi4"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.alloca i64 {bindc_name = "i8", uniq_name = "_QFEi8"}
    %3 = fircg.ext_declare %2 {uniq_name = "_QFEi8"} : (!fir.ref<i64>) -> !fir.ref<i64>
    %4 = fir.alloca !fir.logical<1> {bindc_name = "l1", uniq_name = "_QFEl1"}
    %5 = fircg.ext_declare %4 {uniq_name = "_QFEl1"} : (!fir.ref<!fir.logical<1>>) -> !fir.ref<!fir.logical<1>>
    %6 = fir.alloca !fir.logical<4> {bindc_name = "l4", uniq_name = "_QFEl4"}
    %7 = fircg.ext_declare %6 {uniq_name = "_QFEl4"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
    %8 = fir.alloca f32 {bindc_name = "r4", uniq_name = "_QFEr4"}
    %9 = fircg.ext_declare %8 {uniq_name = "_QFEr4"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %10 = fir.alloca f64 {bindc_name = "r8", uniq_name = "_QFEr8"}
    %11 = fircg.ext_declare %10 {uniq_name = "_QFEr8"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %12 = fir.call @_QFPfn1(%1, %11, %5) fastmath<contract> : (!fir.ref<i32>, !fir.ref<f64>, !fir.ref<!fir.logical<1>>) -> i64
    fir.store %12 to %3 : !fir.ref<i64>
    %13 = fir.call @_QFPfn2(%3, %9, %7) fastmath<contract> : (!fir.ref<i64>, !fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> i32
    fir.store %13 to %1 : !fir.ref<i32>
    return
  } loc(#loc1)
  func.func private @_QFPfn1(%arg0: !fir.ref<i32> {fir.bindc_name = "a"} , %arg1: !fir.ref<f64> {fir.bindc_name = "b"}, %arg2: !fir.ref<!fir.logical<1>> {fir.bindc_name = "c"}) -> i64 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFFfn1Ea"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %2 = fircg.ext_declare %arg1 dummy_scope %0 {uniq_name = "_QFFfn1Eb"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
    %3 = fircg.ext_declare %arg2 dummy_scope %0 {uniq_name = "_QFFfn1Ec"} : (!fir.ref<!fir.logical<1>>, !fir.dscope) -> !fir.ref<!fir.logical<1>>
    %4 = fir.alloca i64 {bindc_name = "res", uniq_name = "_QFFfn1Eres"}
    %5 = fircg.ext_declare %4 {uniq_name = "_QFFfn1Eres"} : (!fir.ref<i64>) -> !fir.ref<i64>
    %6 = fir.load %1 : !fir.ref<i32>
    %7 = fir.convert %6 : (i32) -> f64
    %8 = fir.load %2 : !fir.ref<f64>
    %9 = arith.addf %7, %8 fastmath<contract> : f64
    %10 = fir.convert %9 : (f64) -> i64
    fir.store %10 to %5 : !fir.ref<i64>
    %11 = fir.load %5 : !fir.ref<i64>
    return %11 : i64
  } loc(#loc2)
  func.func private @_QFPfn2(%arg0: !fir.ref<i64> {fir.bindc_name = "a"}, %arg1: !fir.ref<f32> {fir.bindc_name = "b"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "c"}) -> i32 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFFfn2Ea"} : (!fir.ref<i64>, !fir.dscope) -> !fir.ref<i64>
    %2 = fircg.ext_declare %arg1 dummy_scope %0 {uniq_name = "_QFFfn2Eb"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
    %3 = fircg.ext_declare %arg2 dummy_scope %0 {uniq_name = "_QFFfn2Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
    %4 = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFFfn2Eres"}
    %5 = fircg.ext_declare %4 {uniq_name = "_QFFfn2Eres"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %6 = fir.load %1 : !fir.ref<i64>
    %7 = fir.convert %6 : (i64) -> f32
    %8 = fir.load %2 : !fir.ref<f32>
    %9 = arith.addf %7, %8 fastmath<contract> : f32
    %10 = fir.convert %9 : (f32) -> i32
    fir.store %10 to %5 : !fir.ref<i32>
    %11 = fir.load %5 : !fir.ref<i32>
    return %11 : i32
  } loc(#loc3)
  func.func private @_QFPfn3(%arg0: !fir.ref<i32> {fir.bindc_name = "abc"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFFfn3Eabc"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    return
  } loc(#loc4)
}
#loc1 = loc("test.f90":15:1)
#loc2 = loc("test.f90":26:22)
#loc3 = loc("test2.f90":43:22)
#loc4 = loc("test2.f90":53:22)


// CHECK-DAG: #[[INT8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 64, encoding = DW_ATE_signed>
// CHECK-DAG: #[[INT4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK-DAG: #[[REAL8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 64, encoding = DW_ATE_float>
// CHECK-DAG: #[[LOG1:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 8, encoding = DW_ATE_boolean>
// CHECK-DAG: #[[REAL4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #[[LOG4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 32, encoding = DW_ATE_boolean>
// CHECK: #[[TY0:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_program, types = #di_null_type>
// CHECK: #[[TY1:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[INT8]], #[[INT4]], #[[REAL8]], #[[LOG1]]>
// CHECK: #[[TY2:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[INT4]], #[[INT8]], #[[REAL4]], #[[LOG4]]>
// CHECK: #[[TY3:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_null_type, #[[INT4]]>

// Line numbers should match the number in corresponding loc entry.
// CHECK: #llvm.di_subprogram<{{.*}}name = "mn", linkageName = "_QQmain", file = {{.*}}, line = 15, scopeLine = 15, subprogramFlags = "Definition|MainSubprogram", type = #[[TY0]]>
// CHECK: #llvm.di_subprogram<{{.*}}name = "fn1", linkageName = "_QFPfn1", file = {{.*}}, line = 26, scopeLine = 26, subprogramFlags = Definition, type = #[[TY1]]>
// CHECK: #llvm.di_subprogram<{{.*}}name = "fn2", linkageName = "_QFPfn2", file = {{.*}}, line = 43, scopeLine = 43, subprogramFlags = Definition, type = #[[TY2]]>
// CHECK: #llvm.di_subprogram<{{.*}}name = "fn3", linkageName = "_QFPfn3", file = {{.*}}, line = 53, scopeLine = 53, subprogramFlags = Definition, type = #[[TY3]]>

