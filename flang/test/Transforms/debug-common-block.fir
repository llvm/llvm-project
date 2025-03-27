// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  fir.global @__BLNK__ {alignment = 4 : i64} : tuple<i32, !fir.array<8xi8>> {} loc(#loc1)
  fir.global @a_ {alignment = 4 : i64} : tuple<i32, !fir.array<8xi8>> {} loc(#loc2)
  func.func @f1() {
    %c9_i32 = arith.constant 9 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 4.400000e+00 : f32
    %cst_0 = arith.constant 3.300000e+00 : f32
    %cst_1 = arith.constant 2.200000e+00 : f32
    %cst_2 = arith.constant 1.100000e+00 : f32
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@__BLNK__) : !fir.ref<tuple<i32, !fir.array<8xi8>>>
    %1 = fir.convert %0 : (!fir.ref<tuple<i32, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<?xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<f32>
    %4 = fircg.ext_declare %3 {uniq_name = "_QFf1Ex"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc4)
    %5 = fir.address_of(@a_) : !fir.ref<tuple<i32, !fir.array<8xi8>>>
    %6 = fir.convert %5 : (!fir.ref<tuple<i32, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<?xi8>>
    %7 = fir.coordinate_of %6, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %8 = fir.convert %7 : (!fir.ref<i8>) -> !fir.ref<f32>
    %9 = fircg.ext_declare %8 {uniq_name = "_QFf1Exa"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc5)
    %10 = fir.coordinate_of %1, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %11 = fir.convert %10 : (!fir.ref<i8>) -> !fir.ref<f32>
    %12 = fircg.ext_declare %11 {uniq_name = "_QFf1Ey"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc6)
    %13 = fir.coordinate_of %6, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %14 = fir.convert %13 : (!fir.ref<i8>) -> !fir.ref<f32>
    %15 = fircg.ext_declare %14 {uniq_name = "_QFf1Eya"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc7)
    return
  } loc(#loc3)
  func.func @f2() {
    %c16_i32 = arith.constant 16 : i32
    %c6_i32 = arith.constant 6 : i32
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@__BLNK__) : !fir.ref<tuple<i32, !fir.array<8xi8>>> loc(#loc19)
    %1 = fir.convert %0 : (!fir.ref<tuple<i32, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<?xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<f32>
    %4 = fircg.ext_declare %3 {uniq_name = "_QFf2Ex"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc9)
    %5 = fir.address_of(@a_) : !fir.ref<tuple<i32, !fir.array<8xi8>>>
    %6 = fir.convert %5 : (!fir.ref<tuple<i32, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<?xi8>>
    %7 = fir.coordinate_of %6, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %8 = fir.convert %7 : (!fir.ref<i8>) -> !fir.ref<f32>
    %9 = fircg.ext_declare %8 {uniq_name = "_QFf2Exa"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc10)
    %10 = fir.coordinate_of %1, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %11 = fir.convert %10 : (!fir.ref<i8>) -> !fir.ref<f32>
    %12 = fircg.ext_declare %11 {uniq_name = "_QFf2Ey"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc11)
    %13 = fir.coordinate_of %6, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %14 = fir.convert %13 : (!fir.ref<i8>) -> !fir.ref<f32>
    %15 = fircg.ext_declare %14 {uniq_name = "_QFf2Eya"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc12)
    %16 = fir.coordinate_of %1, %c8 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %17 = fir.convert %16 : (!fir.ref<i8>) -> !fir.ref<f32>
    %18 = fircg.ext_declare %17 {uniq_name = "_QFf2Ez"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc13)
    %19 = fir.coordinate_of %6, %c8 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %20 = fir.convert %19 : (!fir.ref<i8>) -> !fir.ref<f32>
    %21 = fircg.ext_declare %20 {uniq_name = "_QFf2Eza"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc14)
    return
  } loc(#loc8)
  func.func @f3() {
    %c24_i32 = arith.constant 24 : i32
    %c23_i32 = arith.constant 23 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@__BLNK__) : !fir.ref<tuple<i32, !fir.array<8xi8>>>
    %1 = fir.convert %0 : (!fir.ref<tuple<i32, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<?xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<i32>
    %4 = fircg.ext_declare %3 {uniq_name = "_QFf3Ex"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc16)
    %5 = fir.address_of(@a_) : !fir.ref<tuple<i32, !fir.array<8xi8>>>
    %6 = fir.convert %5 : (!fir.ref<tuple<i32, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<?xi8>>
    %7 = fir.coordinate_of %6, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %8 = fir.convert %7 : (!fir.ref<i8>) -> !fir.ref<i32>
    %9 = fircg.ext_declare %8 {uniq_name = "_QFf3Exa"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc17)
    return
  } loc(#loc15)
  func.func @test() {
    %c34_i32 = arith.constant 34 : i32
    %c6_i32 = arith.constant 6 : i32
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@__BLNK__) : !fir.ref<tuple<i32, !fir.array<8xi8>>>
    %1 = fir.convert %0 : (!fir.ref<tuple<i32, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<?xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<f32>
    %4 = fircg.ext_declare %3 {uniq_name = "_QFEv1"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc19)
    %5 = fir.coordinate_of %1, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %6 = fir.convert %5 : (!fir.ref<i8>) -> !fir.ref<f32>
    %7 = fircg.ext_declare %6 {uniq_name = "_QFEv2"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc20)
    %8 = fir.coordinate_of %1, %c8 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %9 = fir.convert %8 : (!fir.ref<i8>) -> !fir.ref<f32>
    %10 = fircg.ext_declare %9 {uniq_name = "_QFEv3"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc21)
    %11 = fir.address_of(@a_) : !fir.ref<tuple<i32, !fir.array<8xi8>>>
    %12 = fir.convert %11 : (!fir.ref<tuple<i32, !fir.array<8xi8>>>) -> !fir.ref<!fir.array<?xi8>>
    %13 = fir.coordinate_of %12, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %14 = fir.convert %13 : (!fir.ref<i8>) -> !fir.ref<f32>
    %15 = fircg.ext_declare %14 {uniq_name = "_QFEva1"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc22)
    %16 = fir.coordinate_of %12, %c4 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %17 = fir.convert %16 : (!fir.ref<i8>) -> !fir.ref<f32>
    %18 = fircg.ext_declare %17 {uniq_name = "_QFEva2"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc23)
    %19 = fir.coordinate_of %12, %c8 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %20 = fir.convert %19 : (!fir.ref<i8>) -> !fir.ref<f32>
    %21 = fircg.ext_declare %20 {uniq_name = "_QFEva3"} : (!fir.ref<f32>) -> !fir.ref<f32> loc(#loc24)
    return
  } loc(#loc18)
}

#loc1 = loc(unknown)
#loc2 = loc(unknown)
#loc3 = loc("common.f90":10:1)
#loc4 = loc("common.f90":12:19)
#loc5 = loc("common.f90":12:25)
#loc6 = loc("common.f90":12:22)
#loc7 = loc("common.f90":12:29)
#loc8 = loc("common.f90":20:3)
#loc9 = loc("common.f90":22:3)
#loc10 = loc("common.f90":22:6)
#loc11 = loc("common.f90":22:9)
#loc12 = loc("common.f90":22:13)
#loc13 = loc("common.f90":22:16)
#loc14 = loc("common.f90":22:19)
#loc15 = loc("common.f90":32:18)
#loc16 = loc("common.f90":35:7)
#loc17 = loc("common.f90":35:10)
#loc18 = loc("common.f90":40:1)
#loc19 = loc("common.f90":43:19)
#loc20 = loc("common.f90":43:28)
#loc21 = loc("common.f90":43:22)
#loc22 = loc("common.f90":43:32)
#loc23 = loc("common.f90":43:25)
#loc24 = loc("common.f90":43:36)


// CHECK-DAG: #[[XF1:.*]] = #llvm.di_global_variable<scope = #[[CBF1:.+]], name = "x", linkageName = "_QFf1Ex"{{.*}}>
// CHECK-DAG: #[[YF1:.*]] = #llvm.di_global_variable<scope = #[[CBF1]], name = "y", linkageName = "_QFf1Ey"{{.*}}>
// CHECK-DAG: #[[EXPXF1:.*]] = #llvm.di_global_variable_expression<var = #[[XF1]]>
// CHECK-DAG: #[[EXPYF1:.*]] = #llvm.di_global_variable_expression<var = #[[YF1]], expr = <[DW_OP_plus_uconst(4)]>>
// CHECK-DAG: #[[XAF1:.*]] = #llvm.di_global_variable<scope = #[[CBAF1:.+]], name = "xa", linkageName = "_QFf1Exa"{{.*}}>
// CHECK-DAG: #[[YAF1:.*]] = #llvm.di_global_variable<scope = #[[CBAF1]], name = "ya", linkageName = "_QFf1Eya"{{.*}}>
// CHECK-DAG: #[[EXPXAF1:.*]] = #llvm.di_global_variable_expression<var = #[[XAF1]]>
// CHECK-DAG: #[[EXPYAF1:.*]] = #llvm.di_global_variable_expression<var = #[[YAF1]], expr = <[DW_OP_plus_uconst(4)]>>

// CHECK-DAG: #[[XF2:.*]] = #llvm.di_global_variable<scope = #[[CBF2:.+]], name = "x", linkageName = "_QFf2Ex"{{.*}}>
// CHECK-DAG: #[[YF2:.*]] = #llvm.di_global_variable<scope = #[[CBF2]], name = "y", linkageName = "_QFf2Ey"{{.*}}>
// CHECK-DAG: #[[ZF2:.*]] = #llvm.di_global_variable<scope = #[[CBF2]], name = "z", linkageName = "_QFf2Ez"{{.*}}>
// CHECK-DAG: #[[EXPXF2:.*]] = #llvm.di_global_variable_expression<var = #[[XF2]]>
// CHECK-DAG: #[[EXPYF2:.*]] = #llvm.di_global_variable_expression<var = #[[YF2]], expr = <[DW_OP_plus_uconst(4)]>>
// CHECK-DAG: #[[EXPZF2:.*]] = #llvm.di_global_variable_expression<var = #[[ZF2]], expr = <[DW_OP_plus_uconst(8)]>>
// CHECK-DAG: #[[XAF2:.*]] = #llvm.di_global_variable<scope = #[[CBAF2:.+]], name = "xa", linkageName = "_QFf2Exa"{{.*}}>
// CHECK-DAG: #[[YAF2:.*]] = #llvm.di_global_variable<scope = #[[CBAF2]], name = "ya", linkageName = "_QFf2Eya"{{.*}}>
// CHECK-DAG: #[[ZAF2:.*]] = #llvm.di_global_variable<scope = #[[CBAF2]], name = "za", linkageName = "_QFf2Eza"{{.*}}>
// CHECK-DAG: #[[EXPXAF2:.*]] = #llvm.di_global_variable_expression<var = #[[XAF2]]>
// CHECK-DAG: #[[EXPYAF2:.*]] = #llvm.di_global_variable_expression<var = #[[YAF2]], expr = <[DW_OP_plus_uconst(4)]>>
// CHECK-DAG: #[[EXPZAF2:.*]] = #llvm.di_global_variable_expression<var = #[[ZAF2]], expr = <[DW_OP_plus_uconst(8)]>>

// CHECK-DAG: #[[XF3:.*]] = #llvm.di_global_variable<scope = #[[CBF3:.+]], name = "x", linkageName = "_QFf3Ex"{{.*}}>
// CHECK-DAG: #[[EXPXF3:.*]] = #llvm.di_global_variable_expression<var = #[[XF3]]>
// CHECK-DAG: #[[XAF3:.*]] = #llvm.di_global_variable<scope = #[[CBAF3:.+]], name = "xa", linkageName = "_QFf3Exa"{{.*}}>
// CHECK-DAG: #[[EXPXAF3:.*]] = #llvm.di_global_variable_expression<var = #[[XAF3]]>

// CHECK-DAG: #[[V1:.*]] = #llvm.di_global_variable<scope = #[[CBM:.+]], name = "v1", linkageName = "_QFEv1"{{.*}}>
// CHECK-DAG: #[[EXPV1:.*]] = #llvm.di_global_variable_expression<var = #[[V1]]>
// CHECK-DAG: #[[V2:.*]] = #llvm.di_global_variable<scope = #[[CBM]], name = "v2", linkageName = "_QFEv2"{{.*}}>
// CHECK-DAG: #[[EXPV2:.*]] = #llvm.di_global_variable_expression<var = #[[V2]], expr = <[DW_OP_plus_uconst(4)]>>
// CHECK-DAG: #[[V3:.*]] = #llvm.di_global_variable<scope = #[[CBM]], name = "v3", linkageName = "_QFEv3"{{.*}}>
// CHECK-DAG: #[[EXPV3:.*]] = #llvm.di_global_variable_expression<var = #[[V3]], expr = <[DW_OP_plus_uconst(8)]>>
// CHECK-DAG: #[[VA1:.*]] = #llvm.di_global_variable<scope = #[[CBAM:.+]], name = "va1", linkageName = "_QFEva1"{{.*}}>
// CHECK-DAG: #[[EXPVA1:.*]] = #llvm.di_global_variable_expression<var = #[[VA1]]>
// CHECK-DAG: #[[VA2:.*]] = #llvm.di_global_variable<scope = #[[CBAM]], name = "va2", linkageName = "_QFEva2"{{.*}}>
// CHECK-DAG: #[[EXPVA2:.*]] = #llvm.di_global_variable_expression<var = #[[VA2]], expr = <[DW_OP_plus_uconst(4)]>>
// CHECK-DAG: #[[VA3:.*]] = #llvm.di_global_variable<scope = #[[CBAM]], name = "va3", linkageName = "_QFEva3"{{.*}}>
// CHECK-DAG: #[[EXPVA3:.*]] = #llvm.di_global_variable_expression<var = #[[VA3]], expr = <[DW_OP_plus_uconst(8)]>>

// CHECK-DAG: #[[F1:.*]] = #llvm.di_subprogram<{{.*}}name = "f1"{{.*}}>
// CHECK-DAG: #[[F2:.*]] = #llvm.di_subprogram<{{.*}}name = "f2"{{.*}}>
// CHECK-DAG: #[[F3:.*]] = #llvm.di_subprogram<{{.*}}name = "f3"{{.*}}>
// CHECK-DAG: #[[TEST:.*]] = #llvm.di_subprogram<{{.*}}name = "test"{{.*}}>

// CHECK-DAG: #[[CBF1]] = #llvm.di_common_block<scope = #[[F1]], name = "__BLNK__"{{.*}}>
// CHECK-DAG: #[[CBF2]] = #llvm.di_common_block<scope = #[[F2]], name = "__BLNK__"{{.*}}>
// CHECK-DAG: #[[CBF3]] = #llvm.di_common_block<scope = #[[F3]], name = "__BLNK__"{{.*}}>
// CHECK-DAG: #[[CBM]] = #llvm.di_common_block<scope = #[[TEST]], name = "__BLNK__"{{.*}}>
// CHECK-DAG: #[[CBAF1]] = #llvm.di_common_block<scope = #[[F1]], name = "a"{{.*}}>
// CHECK-DAG: #[[CBAF2]] = #llvm.di_common_block<scope = #[[F2]], name = "a"{{.*}}>
// CHECK-DAG: #[[CBAF3]] = #llvm.di_common_block<scope = #[[F3]], name = "a"{{.*}}>
// CHECK-DAG: #[[CBAM]] = #llvm.di_common_block<scope = #[[TEST]], name = "a"{{.*}}>

// CHECK-DAG: fir.global @__BLNK__ {{.*}} loc(#[[LOC1:.*]])
// CHECK-DAG: fir.global @a_ {{.*}} loc(#[[LOC2:.*]])
// CHECK-DAG: #[[LOC1]]
// CHECK-DAG-SAME: #[[EXPXF1]]
// CHECK-DAG-SAME: #[[EXPYF1]]
// CHECK-DAG-SAME: #[[EXPXF2]]
// CHECK-DAG-SAME: #[[EXPYF2]]
// CHECK-DAG-SAME: #[[EXPZF2]]
// CHECK-DAG-SAME: #[[EXPXF3]]
// CHECK-DAG-SAME: #[[EXPV1]]
// CHECK-DAG-SAME: #[[EXPV2]]
// CHECK-DAG-SAME: #[[EXPV3]]
// CHECK-DAG: #[[LOC2]]
// CHECK-DAG-SAME: #[[EXPXAF1]]
// CHECK-DAG-SAME: #[[EXPYAF1]]
// CHECK-DAG-SAME: #[[EXPXAF2]]
// CHECK-DAG-SAME: #[[EXPYAF2]]
// CHECK-DAG-SAME: #[[EXPZAF2]]
// CHECK-DAG-SAME: #[[EXPXAF3]]
// CHECK-DAG-SAME: #[[EXPVA1]]
// CHECK-DAG-SAME: #[[EXPVA2]]
// CHECK-DAG-SAME: #[[EXPVA3]]
