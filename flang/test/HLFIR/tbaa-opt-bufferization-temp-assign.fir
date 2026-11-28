// RUN: fir-opt --split-input-file --opt-bufferization --convert-hlfir-to-fir --fir-add-alias-tags %s | FileCheck %s

// Bufferization must preserve the destination's dummy scope when redirecting
// the constructor stores. Trace the tags to their roots to distinguish the
// destination's scope from the intervening decoy scope.
// CHECK-DAG: #[[DECOY_ROOT:.*]] = #llvm.tbaa_root<id = "Flang function root array_section_constructor - Scope 1">
// CHECK-DAG: #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang function root array_section_constructor">
// CHECK-DAG: #[[DECOY_ACCESS:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[DECOY_ROOT]], 0>}>
// CHECK-DAG: #[[ACCESS:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK-DAG: #[[DECOY_DATA:.*]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[DECOY_ACCESS]], 0>}>
// CHECK-DAG: #[[DATA:.*]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ACCESS]], 0>}>
// CHECK-DAG: #[[DECOY_ARGS:.*]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[DECOY_DATA]], 0>}>
// CHECK-DAG: #[[ALLOCATED:.*]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[DATA]], 0>}>
// CHECK-DAG: #[[ARGS:.*]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[DATA]], 0>}>
// CHECK-DAG: #[[DECOY_TYPE:.*]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFarray_section_constructorEdecoy", members = {<#[[DECOY_ARGS]], 0>}>
// CHECK-DAG: #[[DEST_TYPE:.*]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFarray_section_constructorEsinc", members = {<#[[ARGS]], 0>}>
// CHECK-DAG: #[[INDEX_TAG:.*]] = #llvm.tbaa_tag<base_type = #[[ALLOCATED]], access_type = #[[ALLOCATED]], offset = 0>
// CHECK-DAG: #[[DECOY_TAG:.*]] = #llvm.tbaa_tag<base_type = #[[DECOY_TYPE]], access_type = #[[DECOY_TYPE]], offset = 0>
// CHECK-DAG: #[[DEST_TAG:.*]] = #llvm.tbaa_tag<base_type = #[[DEST_TYPE]], access_type = #[[DEST_TYPE]], offset = 0>
// CHECK-LABEL: func.func @array_section_constructor(
// CHECK: %[[INDEX:.*]] = fir.alloca index
// CHECK: %[[SCOPE:.*]] = fir.dummy_scope
// CHECK: %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
// CHECK: %[[DEST:.*]] = fir.declare {{.*}} dummy_scope %[[SCOPE]] arg 2
// CHECK: %[[DECOY_SCOPE:.*]] = fir.dummy_scope
// CHECK: %[[DECOY:.*]] = fir.declare {{.*}} dummy_scope %[[DECOY_SCOPE]] uniq_name("_QFarray_section_constructorEdecoy")
// CHECK: fir.store %{{.*}} to %[[DECOY]] <{tbaa = [#[[DECOY_TAG]]]}> : !fir.ref<index>
// CHECK: fir.store %{{.*}} to %[[INDEX]] <{tbaa = [#[[INDEX_TAG]]]}> : !fir.ref<index>
// CHECK: %[[EMBOX:.*]] = fir.embox %[[DEST]](%[[SHAPE]]) : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<4xf64>>
// CHECK: %[[SHIFT:.*]] = fir.shift %c1 : (index) -> !fir.shift<1>
// CHECK: %[[REBOX:.*]] = fir.rebox %[[EMBOX]](%[[SHIFT]]) : (!fir.box<!fir.array<4xf64>>, !fir.shift<1>) -> !fir.box<!fir.array<4xf64>>
// CHECK: fir.do_loop
// CHECK: fir.store %{{.*}} to %[[INDEX]] <{tbaa = [#[[INDEX_TAG]]]}> : !fir.ref<index>
// CHECK: %[[ELEMENT:.*]] = fir.array_coor %[[REBOX]](
// CHECK: fir.store %{{.*}} to %[[ELEMENT]] <{tbaa = [#[[DEST_TAG]]]}> : !fir.ref<f64>

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
func.func @array_section_constructor(%arg0: !fir.ref<!fir.array<4xf64>> {fir.bindc_name = "stress", fir.fortran_attrs = #fir.var_attrs<intent_in>}, %arg1: !fir.ref<!fir.array<4xf64>> {fir.bindc_name = "sinc", fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %true = arith.constant true
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = fir.alloca index
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.shape %c4 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %1 arg 2 uniq_name("_QFarray_section_constructorEsinc") fortran_attrs<intent_out> : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
  %4:2 = hlfir.declare %arg0(%2) dummy_scope %1 arg 1 uniq_name("_QFarray_section_constructorEstress") fortran_attrs<intent_in> : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)

  %decoy_scope = fir.dummy_scope : !fir.dscope
  %decoy_storage = fir.alloca index
  %decoy:2 = hlfir.declare %decoy_storage dummy_scope %decoy_scope uniq_name("_QFarray_section_constructorEdecoy") : (!fir.ref<index>, !fir.dscope) -> (!fir.ref<index>, !fir.ref<index>)
  fir.store %c1 to %decoy#0 : !fir.ref<index>

  fir.store %c1 to %0 : !fir.ref<index>
  %5 = fir.allocmem !fir.array<4xf64> <{bindc_name = ".tmp.arrayctor", uniq_name = ""}>
  %6:2 = hlfir.declare %5(%2) uniq_name(".tmp.arrayctor") : (!fir.heap<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<4xf64>>, !fir.heap<!fir.array<4xf64>>)
  %7 = hlfir.designate %4#0 (%c1:%c4:%c1)  shape %2 : (!fir.ref<!fir.array<4xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  fir.do_loop %arg2 = %c1 to %c4 step %c1 {
    %9 = hlfir.designate %7 (%arg2)  : (!fir.ref<!fir.array<4xf64>>, index) -> !fir.ref<f64>
    %10 = fir.load %9 : !fir.ref<f64>
    %11 = fir.load %0 : !fir.ref<index>
    %12 = arith.addi %11, %c1 : index
    fir.store %12 to %0 : !fir.ref<index>
    %13 = hlfir.designate %6#0 (%11)  : (!fir.heap<!fir.array<4xf64>>, index) -> !fir.ref<f64>
    hlfir.assign %10 to %13 : f64, !fir.ref<f64>
  }
  %8 = hlfir.as_expr %6#0 move %true : (!fir.heap<!fir.array<4xf64>>, i1) -> !hlfir.expr<4xf64>
  hlfir.assign %8 to %3#0 : !hlfir.expr<4xf64>, !fir.ref<!fir.array<4xf64>>
  hlfir.destroy %8 : !hlfir.expr<4xf64>
  return
}
}
