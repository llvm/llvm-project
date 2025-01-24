// RUN: fir-opt --function-attr %s | FileCheck %s

// If a function has a body and is not bind(c), and if the dummy argument doesn't have the target,
// asynchronous, volatile, or pointer attribute, then add llvm.nocapture to the dummy argument.

func.func @_QParg_nocapture(%arg0: !fir.ref<i32> {fir.bindc_name = "tar", fir.target}, %arg1: !fir.ref<i32> {fir.asynchronous, fir.bindc_name = "asynch"}, %arg2: !fir.ref<i32> {fir.bindc_name = "vol", fir.volatile}, %arg3: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "ptr"}, %arg4: !fir.ref<i32> {fir.bindc_name = "nocap"}) {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFarg_nocaptureEtar"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %2 = fir.declare %arg1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<asynchronous>, uniq_name = "_QFarg_nocaptureEasynch"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %3 = fir.declare %arg2 dummy_scope %0 {uniq_name = "_QFarg_nocaptureEvol"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %4 = fir.declare %arg3 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFarg_nocaptureEptr"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
    %5 = fir.declare %arg4 dummy_scope %0 {uniq_name = "_QFarg_nocaptureEnocap"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    return
}
// CHECK-LABEL: func.func @_QParg_nocapture(
// CHECK-SAME:                                 %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "tar", fir.target},
// CHECK-SAME:                                 %[[ARG1:.*]]: !fir.ref<i32> {fir.asynchronous, fir.bindc_name = "asynch"},
// CHECK-SAME:                                 %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "vol", fir.volatile},
// CHECK-SAME:                                 %[[ARG3:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "ptr", llvm.nocapture},
// CHECK-SAME:                                 %[[ARG4:.*]]: !fir.ref<i32> {fir.bindc_name = "nocap", llvm.nocapture}) {
// CHECK:    return
// CHECK-NEXT: }

func.func @arg_nocapture_bindc(%arg0: !fir.ref<i32> {fir.bindc_name = "tar", fir.target}, %arg1: !fir.ref<i32> {fir.bindc_name = "nocap"}) attributes {fir.bindc_name = "arg_nocapture_bindc", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFarg_nocapture_bindcEtar"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFarg_nocapture_bindcEnocap"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    return
}
// CHECK-LABEL: func.func @arg_nocapture_bindc(
// CHECK-NOT:  llvm.nocapture


// If a function declaration is from a module and is not bind(c), and if the dummy argument doesn't have
// the target, asynchronous, volatile, or pointer attribute, then add llvm.nocapture to the dummy argument.

func.func private @_QMarg_modPcheck_args(!fir.ref<i32> {fir.target}, !fir.ref<i32> {fir.asynchronous}, !fir.ref<i32> {fir.volatile}, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<i32>, !fir.boxchar<1>, !fir.ref<complex<f32>>)
// CHECK-LABEL: func.func private @_QMarg_modPcheck_args(
// CHECK-SAME:                                 !fir.ref<i32> {fir.target},
// CHECK-SAME:                                 !fir.ref<i32> {fir.asynchronous},
// CHECK-SAME:                                 !fir.ref<i32> {fir.volatile},
// CHECK-SAME:                                 !fir.ref<!fir.box<!fir.ptr<i32>>> {llvm.nocapture},
// CHECK-SAME:                                 !fir.ref<i32> {llvm.nocapture},
// CHECK-SAME:                                 !fir.boxchar<1>,
// CHECK-SAME:                                 !fir.ref<complex<f32>> {llvm.nocapture})
