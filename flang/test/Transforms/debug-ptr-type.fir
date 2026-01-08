// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module {
  fir.global @_QMhelperEpar : !fir.box<!fir.ptr<!fir.array<?x?xf32>>> {
    %0 = fir.zero_bits !fir.ptr<!fir.array<?x?xf32>>
    %c0 = arith.constant 0 : index
    %1 = fir.shape %c0, %c0 : (index, index) -> !fir.shape<2>
    %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
    fir.has_value %2 : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  } loc(#loc1)
  fir.global @_QMhelperEpar2 : !fir.box<!fir.ptr<!fir.array<?xi32>>> {
    %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
    %c0 = arith.constant 0 : index
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
    fir.has_value %2 : !fir.box<!fir.ptr<!fir.array<?xi32>>>
  } loc(#loc2)
  fir.global @_QMhelperEpsc : !fir.box<!fir.ptr<i32>> {
    %0 = fir.zero_bits !fir.ptr<i32>
    %1 = fir.embox %0 : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
    fir.has_value %1 : !fir.box<!fir.ptr<i32>>
  } loc(#loc3)
  fir.global @_QMmEpstr : !fir.box<!fir.ptr<!fir.char<1,16>>> {
    %0 = fir.zero_bits !fir.ptr<!fir.char<1,16>>
    %1 = fir.embox %0 : (!fir.ptr<!fir.char<1,16>>) -> !fir.box<!fir.ptr<!fir.char<1,16>>>
    fir.has_value %1 : !fir.box<!fir.ptr<!fir.char<1,16>>>
  } loc(#loc4)
}
#loc1 = loc("test.f90":5:1)
#loc2 = loc("test.f90":6:1)
#loc3 = loc("test.f90":7:1)
#loc4 = loc("test.f90":8:1)

// CHECK-DAG: #[[INT_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer"{{.*}}>
// CHECK-DAG: #[[ARR1_TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}dataLocation = {{.*}}, associated = <[DW_OP_push_object_address, DW_OP_deref, DW_OP_lit0, DW_OP_ne]>, elements = #llvm.di_subrange<count = #llvm.di_expression<{{.*}}>, lowerBound = #llvm.di_expression<{{.*}}>, #llvm.di_subrange<count = #llvm.di_expression<{{.*}}>, lowerBound = #llvm.di_expression<{{.*}}>>>
// CHECK-DAG: #[[ARR2_TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}dataLocation = {{.*}}, associated = <[DW_OP_push_object_address, DW_OP_deref, DW_OP_lit0, DW_OP_ne]>, elements = #llvm.di_subrange<count = #llvm.di_expression<{{.*}}>, lowerBound = #llvm.di_expression<{{.*}}>>>

// CHECK-DAG: #[[PTR_TY:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type{{.*}}baseType = #[[INT_TY]]{{.*}}>
// CHECK-DAG: #llvm.di_global_variable<{{.*}}name = "par"{{.*}}type = #[[ARR1_TY]]{{.*}}>
// CHECK-DAG: #llvm.di_global_variable<{{.*}}name = "par2"{{.*}}type = #[[ARR2_TY]]{{.*}}>
// CHECK-DAG: #llvm.di_global_variable<{{.*}}name = "psc"{{.*}}type = #[[PTR_TY]]{{.*}}>
