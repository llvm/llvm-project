// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QQmain() attributes {fir.bindc_name = "mn"} {
    %0 = fir.alloca i32 {bindc_name = "i4", uniq_name = "_QFEi4"}
    %1 = fircg.ext_declare %0 {uniq_name = "_QFEi4"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc1)
    %2 = fir.alloca i64 {bindc_name = "i8", uniq_name = "_QFEi8"}
    %3 = fircg.ext_declare %2 {uniq_name = "_QFEi8"} : (!fir.ref<i64>) -> !fir.ref<i64> loc(#loc2)
    %4 = fir.alloca !fir.logical<1> {bindc_name = "l1", uniq_name = "_QFEl1"}
    %5 = fircg.ext_declare %4 {uniq_name = "_QFEl1"} : (!fir.ref<!fir.logical<1>>) -> !fir.ref<!fir.logical<1>> loc(#loc3)
    %6 = fir.alloca !fir.logical<4> {bindc_name = "l4", uniq_name = "_QFEl4"}
    %7 = fircg.ext_declare %6 {uniq_name = "_QFEl4"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>> loc(#loc4)
    %8 = fir.alloca f32 {bindc_name = "r4", uniq_name = "_QFEr4"}
    %9 = fircg.ext_declare %8 {uniq_name = "_QFEr4"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc5)
    %10 = fir.alloca f64 {bindc_name = "r8", uniq_name = "_QFEr8"}
    %11 = fircg.ext_declare %10 {uniq_name = "_QFEr8"} : (!fir.ref<f64>) -> !fir.ref<f64> loc(#loc6)
    %12 = fir.call @_QFPfn1(%1, %11, %5) fastmath<contract> : (!fir.ref<i32>, !fir.ref<f64>, !fir.ref<!fir.logical<1>>) -> i64
    fir.store %12 to %3 : !fir.ref<i64>
    %13 = fir.call @_QFPfn2(%3, %9, %7) fastmath<contract> : (!fir.ref<i64>, !fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> i32
    fir.store %13 to %1 : !fir.ref<i32>
    return
  } loc(#loc7)
  func.func private @_QFPfn1(%arg0: !fir.ref<i32> {fir.bindc_name = "a1"}, %arg1: !fir.ref<f64> {fir.bindc_name = "b1"}, %arg2: !fir.ref<!fir.logical<1>> {fir.bindc_name = "c1"}) -> i64 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.undefined !fir.dscope loc(#loc11)
    %1 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFFfn1Ea1"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc8)
    %2 = fircg.ext_declare %arg1 dummy_scope %0 {uniq_name = "_QFFfn1Eb1"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64> loc(#loc9)
    %3 = fircg.ext_declare %arg2 dummy_scope %0 {uniq_name = "_QFFfn1Ec1"} : (!fir.ref<!fir.logical<1>>, !fir.dscope) -> !fir.ref<!fir.logical<1>> loc(#loc10)
    %4 = fir.alloca i64 {bindc_name = "res1", uniq_name = "_QFFfn1Eres1"} loc(#loc15)
    %5 = fircg.ext_declare %4 {uniq_name = "_QFFfn1Eres1"} : (!fir.ref<i64>) -> !fir.ref<i64> loc(#loc11)
    %6 = fir.load %1 : !fir.ref<i32>
    %7 = fir.convert %6 : (i32) -> f64
    %8 = fir.load %2 : !fir.ref<f64>
    %9 = arith.addf %7, %8 fastmath<contract> : f64
    %10 = fir.convert %9 : (f64) -> i64
    fir.store %10 to %5 : !fir.ref<i64>
    %11 = fir.load %5 : !fir.ref<i64>
    return %11 : i64
  } loc(#loc12)
  func.func private @_QFPfn2(%arg0: !fir.ref<i64> {fir.bindc_name = "a2"}, %arg1: !fir.ref<f32> {fir.bindc_name = "b2"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "c2"}) -> i32 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFFfn2Ea2"} : (!fir.ref<i64>, !fir.dscope) -> !fir.ref<i64> loc(#loc13)
    %2 = fircg.ext_declare %arg1 dummy_scope %0 {uniq_name = "_QFFfn2Eb2"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32> loc(#loc14)
    %3 = fircg.ext_declare %arg2 dummy_scope %0 {uniq_name = "_QFFfn2Ec2"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>> loc(#loc15)
    %4 = fir.alloca i32 {bindc_name = "res2", uniq_name = "_QFFfn2Eres2"}
    %5 = fircg.ext_declare %4 {uniq_name = "_QFFfn2Eres2"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc16)
    %6 = fir.load %1 : !fir.ref<i64>
    %7 = fir.convert %6 : (i64) -> f32
    %8 = fir.load %2 : !fir.ref<f32>
    %9 = arith.addf %7, %8 fastmath<contract> : f32
    %10 = fir.convert %9 : (f32) -> i32
    fir.store %10 to %5 : !fir.ref<i32>
    %11 = fir.load %5 : !fir.ref<i32>
    return %11 : i32
  } loc(#loc17)
}
#loc7 = loc("test.f90":4:19)
#loc1 = loc("test.f90":5:1)
#loc2 = loc("test.f90":6:22)
#loc3 = loc("test.f90":7:22)
#loc4 = loc("test.f90":8:22)
#loc5 = loc("test.f90":9:22)
#loc6 = loc("test.f90":10:19)
#loc12 = loc("test.f90":12:36)
#loc8 = loc("test.f90":13:3)
#loc9 = loc("test.f90":14:3)
#loc10 = loc("test.f90":15:1)
#loc11 = loc("test.f90":16:1)
#loc17 = loc("test.f90":18:33)
#loc13 = loc("test.f90":19:33)
#loc14 = loc("test.f90":20:36)
#loc15 = loc("test.f90":21:24)
#loc16 = loc("test.f90":22:5)

// CHECK-DAG: #[[INT8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 64, encoding = DW_ATE_signed>
// CHECK-DAG: #[[INT4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK-DAG: #[[REAL8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 64, encoding = DW_ATE_float>
// CHECK-DAG: #[[LOG1:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 8, encoding = DW_ATE_boolean>
// CHECK-DAG: #[[REAL4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #[[LOG4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 32, encoding = DW_ATE_boolean>
// CHECK-DAG: #[[MAIN:.*]] = #llvm.di_subprogram<{{.*}}name = "mn"{{.*}}>
// CHECK-DAG: #[[FN1:.*]] = #llvm.di_subprogram<{{.*}}name = "fn1"{{.*}}>
// CHECK-DAG: #[[FN2:.*]] = #llvm.di_subprogram<{{.*}}name = "fn2"{{.*}}>

// Line numbers should match the number in corresponding loc entry.
// CHECK-DAG: #[[I4:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "i4", file = #{{.*}}, line = 5, type = #[[INT4]]>
// CHECK-DAG: #[[I8:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "i8", file = #{{.*}}, line = 6, type = #[[INT8]]>
// CHECK-DAG: #[[R4:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "r4", file = #{{.*}}, line = 9, type = #[[REAL4]]>
// CHECK-DAG: #[[R8:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "r8", file = #{{.*}}, line = 10, type = #[[REAL8]]>
// CHECK-DAG: #[[L1:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "l1", file = #{{.*}}, line = 7, type = #[[LOG1]]>
// CHECK-DAG: #[[L4:.*]] = #llvm.di_local_variable<scope = #[[MAIN]], name = "l4", file = #{{.*}}, line = 8, type = #[[LOG4]]>

// CHECK-DAG: #[[A1:.*]] = #llvm.di_local_variable<scope = #[[FN1]], name = "a1", file = #{{.*}}, line = 13, arg = 1, type = #[[INT4]]>
// CHECK-DAG: #[[B1:.*]] = #llvm.di_local_variable<scope = #[[FN1]], name = "b1", file = #{{.*}}, line = 14, arg = 2, type = #[[REAL8]]>
// CHECK-DAG: #[[C1:.*]] = #llvm.di_local_variable<scope = #[[FN1]], name = "c1", file = #{{.*}}, line = 15, arg = 3, type = #[[LOG1]]>
// CHECK-DAG: #[[RES1:.*]] = #llvm.di_local_variable<scope = #[[FN1]], name = "res1", file = #{{.*}}, line = 16, type = #[[INT8]]>

// CHECK-DAG: #[[A2:.*]] = #llvm.di_local_variable<scope = #[[FN2]], name = "a2", file = #{{.*}}, line = 19, arg = 1, type = #[[INT8]]>
// CHECK-DAG: #[[B2:.*]] = #llvm.di_local_variable<scope = #[[FN2]], name = "b2", file = #{{.*}}, line = 20, arg = 2, type = #[[REAL4]]>
// CHECK-DAG: #[[C2:.*]] = #llvm.di_local_variable<scope = #[[FN2]], name = "c2", file = #{{.*}}, line = 21, arg = 3, type = #[[LOG4]]>
// CHECK-DAG: #[[RES2:.*]] = #llvm.di_local_variable<scope = #[[FN2]], name = "res2", file = #{{.*}}, line = 22, type = #[[INT4]]>
