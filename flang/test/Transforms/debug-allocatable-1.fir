// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s


module {
  func.func private @_QFPff()  {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = fir.undefined !fir.dscope
    %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "ar2", uniq_name = "_QFFffEar2"}
    %4 = fircg.ext_declare %1 {uniq_name = "_QFFffEar2"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>  loc(#loc1)
    %15 = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "sc", uniq_name = "_QFFffEsc"}
    %18 = fircg.ext_declare %15 {uniq_name = "_QFFffEsc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>> loc(#loc2)
    return
  } loc(#loc3)
}

#loc1 = loc("test.f90":3:3)
#loc2 = loc("test.f90":4:3)
#loc3 = loc("test.f90":1:3)

// CHECK-DAG: #[[TY1:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real"{{.*}}>
// CHECK-DAG: #[[TY2:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}allocated = <[DW_OP_push_object_address, DW_OP_deref, DW_OP_lit0, DW_OP_ne]>{{.*}}elements = #llvm.di_subrange{{.*}}#llvm.di_subrange
// CHECK-DAG: #[[TY3:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type{{.*}}baseType = #[[TY1]]{{.*}}>

// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "ar2"{{.*}}type = #[[TY2]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "sc"{{.*}}type = #[[TY3]]>
