// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QQmain() attributes {fir.bindc_name = "mn"} {
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c-2 = arith.constant -2 : index loc(#loc3)
    %c4 = arith.constant 4 : index loc(#loc3)
    %0 = fir.alloca !fir.array<3xi32> {bindc_name = "d1", uniq_name = "_QFEd1"}
    %1 = fircg.ext_declare %0(%c3) {uniq_name = "_QFEd1"} : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<!fir.array<3xi32>> loc(#loc1)
    %2 = fir.address_of(@_QFEd2) : !fir.ref<!fir.array<2x5xi32>>
    %3 = fircg.ext_declare %2(%c2, %c5) {uniq_name = "_QFEd2"} : (!fir.ref<!fir.array<2x5xi32>>, index, index) -> !fir.ref<!fir.array<2x5xi32>> loc(#loc2)
    %4 = fir.address_of(@_QFEd3) : !fir.ref<!fir.array<6x8x7xf32>>
    %5 = fircg.ext_declare %4(%c6, %c8, %c7) {uniq_name = "_QFEd3"} : (!fir.ref<!fir.array<6x8x7xf32>>, index, index, index) -> !fir.ref<!fir.array<6x8x7xf32>> loc(#loc3)
    %6 = fir.address_of(@_QFEd4) : !fir.ref<!fir.array<6x7xi32>>
    %7 = fircg.ext_declare %6(%c6, %c7) origin %c-2, %c4 {uniq_name = "_QFEd4"} : (!fir.ref<!fir.array<6x7xi32>>, index, index, index, index) -> !fir.ref<!fir.array<6x7xi32>> loc(#loc5)
    return
  } loc(#loc4)
}

#loc1 = loc("test.f90":5:1)
#loc2 = loc("test.f90":6:11)
#loc3 = loc("test.f90":7:11)
#loc4 = loc("test.f90":2:8)
#loc5 = loc("test.f90":8:11)


// CHECK-DAG: #[[INT:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK-DAG: #[[REAL:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #[[D1TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}baseType = #[[INT]], elements = #llvm.di_subrange<count = 3 : i64>>
// CHECK-DAG: #[[D2TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}baseType = #[[INT]], elements = #llvm.di_subrange<count = 2 : i64>, #llvm.di_subrange<count = 5 : i64>>
// CHECK-DAG: #[[D3TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}baseType = #[[REAL]], elements = #llvm.di_subrange<count = 6 : i64>, #llvm.di_subrange<count = 8 : i64>, #llvm.di_subrange<count = 7 : i64>>
// CHECK-DAG: #[[D4TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, elements = #llvm.di_subrange<count = 6 : i64, lowerBound = -2 : i64>, #llvm.di_subrange<count = 7 : i64, lowerBound = 4 : i64>>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "d1"{{.*}}type = #[[D1TY]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "d2"{{.*}}type = #[[D2TY]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "d3"{{.*}}type = #[[D3TY]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "d4"{{.*}}type = #[[D4TY]]>
