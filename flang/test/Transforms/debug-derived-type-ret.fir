// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module {
  func.func @_QMmodPbuild(%arg0: !fir.ref<!fir.type<_QMmodTcart{x1:i32}>> {fir.result_arg}, %arg1: !fir.ref<i32> {fir.bindc_name = "x"}) attributes {fir.internal_name = "_QMmodPbuild"} {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 uniq_name("_QMmodFbuildEbuild") : (!fir.ref<!fir.type<_QMmodTcart{x1:i32}>>) -> !fir.ref<!fir.type<_QMmodTcart{x1:i32}>>
    %2 = fircg.ext_declare %arg1 dummy_scope %0 arg 1 uniq_name("_QMmodFbuildEx") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %3 = fir.load %2 : !fir.ref<i32>
    %4 = fir.coordinate_of %1, x1 : (!fir.ref<!fir.type<_QMmodTcart{x1:i32}>>) -> !fir.ref<i32>
    fir.store %3 to %4 : !fir.ref<i32>
    return
  } loc(#loc1)

  // The declaration of a function defined elsewhere is described the same way.
  func.func private @_QPext(!fir.ref<!fir.type<_QMmodTcart{x1:i32}>> {fir.result_arg}, !fir.ref<i32>) loc(#loc3)

  // An array result is passed the same way.
  func.func @_QMmodParr3(%arg0: !fir.ref<!fir.array<3xi32>> {fir.result_arg}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.internal_name = "_QMmodParr3"} {
    %c3 = arith.constant 3 : index
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg1 dummy_scope %0 arg 1 uniq_name("_QMmodFarr3En") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %2 = fircg.ext_declare %arg0(%c3) uniq_name("_QMmodFarr3Er") : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<!fir.array<3xi32>>
    return
  } loc(#loc2)
}

#loc1 = loc("test.f90":6:1)
#loc2 = loc("test.f90":12:1)
#loc3 = loc("test.f90":18:1)


// CHECK-DAG: #[[CART_TYPE:.*]] = #llvm.di_composite_type<{{.*}}tag = DW_TAG_structure_type, name = "cart"
// CHECK-DAG: #[[INT_TYPE:.*]] = #llvm.di_basic_type<{{.*}}name = "integer(kind=4)"
// CHECK-DAG: #[[SUBR_TYPE:.*]] = #llvm.di_subroutine_type<{{.*}}types = #[[CART_TYPE]], #[[INT_TYPE]]>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "build"{{.*}}type = #[[SUBR_TYPE]]{{[,>]}}
// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "ext"{{.*}}type = #[[SUBR_TYPE]]{{[,>]}}

// CHECK-DAG: #[[ARR_TYPE:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #[[INT_TYPE]],
// CHECK-DAG: #[[ARR_SUBR:.*]] = #llvm.di_subroutine_type<{{.*}}types = #[[ARR_TYPE]], #[[INT_TYPE]]>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "arr3"{{.*}}type = #[[ARR_SUBR]]{{[,>]}}
