// An assignment to an unallocated allocatable scalar stores the value into a
// new buffer before the buffer is stored into the descriptor. That store must
// get the same tag as the accesses through the descriptor of the variable.
//
// module m
//   integer, allocatable, target :: t
// contains
//   subroutine dummy_var(j, n, r)
//     integer, allocatable :: j
//     integer :: n, r
//     j = n
//     r = j
//   end subroutine
//   subroutine save_var(n, r)
//     integer :: n, r
//     integer, allocatable, save :: j
//     j = n
//     r = j
//   end subroutine
//   subroutine module_target(n, r)
//     integer :: n, r
//     t = n
//     r = t
//   end subroutine
//   subroutine local_var(n, r)
//     integer :: n, r
//     integer, allocatable :: j
//     j = n
//     r = j
//   end subroutine
// end module

// RUN: fir-opt --fir-add-alias-tags %s | FileCheck %s

// CHECK-DAG: #[[DUMMY_J:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QMmFdummy_varEj", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[DIRECT_J:.+]] = #llvm.tbaa_type_desc<id = "direct data/_QMmFsave_varEj", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[TARGET_T:.+]] = #llvm.tbaa_type_desc<id = "target data/_QMmEt", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[LOCAL_J:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QMmFlocal_varEj", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[DUMMY_J_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMY_J]], access_type = #[[DUMMY_J]], offset = 0>
// CHECK-DAG: #[[DIRECT_J_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[DIRECT_J]], access_type = #[[DIRECT_J]], offset = 0>
// CHECK-DAG: #[[TARGET_T_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[TARGET_T]], access_type = #[[TARGET_T]], offset = 0>
// CHECK-DAG: #[[LOCAL_J_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[LOCAL_J]], access_type = #[[LOCAL_J]], offset = 0>

module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    !llvm.ptr = dense<64> : vector<4xi64>,
    i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>,
    i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>,
    i64 = dense<[32, 64]> : vector<2xi64>,
    f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>,
    f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">,
  llvm.data_layout = ""} {

// CHECK-LABEL: func.func @_QMmPdummy_var(
// CHECK: fir.store %[[N:.*]] to %{{.*}} <{tbaa = [#[[DUMMY_J_TAG]]]}> : !fir.heap<i32>
// CHECK: %[[NEW:.*]] = fir.allocmem i32
// CHECK: fir.store %[[N]] to %[[NEW]] <{tbaa = [#[[DUMMY_J_TAG]]]}> : !fir.heap<i32>
// CHECK: fir.load %{{.*}} <{tbaa = [#[[DUMMY_J_TAG]]]}> : !fir.heap<i32>
func.func @_QMmPdummy_var(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "j"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "r"}) {
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFdummy_varEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QMmFdummy_varEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 uniq_name("_QMmFdummy_varEr") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %2 : !fir.ref<i32>
  %5 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %6 = fir.box_addr %5 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %7 = fir.convert %6 : (!fir.heap<i32>) -> i64
  %8 = arith.cmpi ne, %7, %c0_i64 : i64
  cf.cond_br %8, ^bb1, ^bb2
^bb1:
  fir.store %4 to %6 : !fir.heap<i32>
  cf.br ^bb3
^bb2:
  %9 = fir.allocmem i32
  fir.store %4 to %9 : !fir.heap<i32>
  %10 = fir.embox %9 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %10 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  cf.br ^bb3
^bb3:
  %11 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %12 = fir.box_addr %11 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %13 = fir.load %12 : !fir.heap<i32>
  fir.store %13 to %3 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @_QMmPsave_var(
// CHECK: fir.store %[[N:.*]] to %{{.*}} <{tbaa = [#[[DIRECT_J_TAG]]]}> : !fir.heap<i32>
// CHECK: %[[NEW:.*]] = fir.allocmem i32
// CHECK: fir.store %[[N]] to %[[NEW]] <{tbaa = [#[[DIRECT_J_TAG]]]}> : !fir.heap<i32>
// CHECK: fir.load %{{.*}} <{tbaa = [#[[DIRECT_J_TAG]]]}> : !fir.heap<i32>
func.func @_QMmPsave_var(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<i32> {fir.bindc_name = "r"}) {
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMmFsave_varEj) : !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %1 uniq_name("_QMmFsave_varEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFsave_varEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QMmFsave_varEr") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %5 = fir.load %3 : !fir.ref<i32>
  %6 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %8 = fir.convert %7 : (!fir.heap<i32>) -> i64
  %9 = arith.cmpi ne, %8, %c0_i64 : i64
  cf.cond_br %9, ^bb1, ^bb2
^bb1:
  fir.store %5 to %7 : !fir.heap<i32>
  cf.br ^bb3
^bb2:
  %10 = fir.allocmem i32
  fir.store %5 to %10 : !fir.heap<i32>
  %11 = fir.embox %10 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %11 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  cf.br ^bb3
^bb3:
  %12 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %13 = fir.box_addr %12 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %14 = fir.load %13 : !fir.heap<i32>
  fir.store %14 to %4 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @_QMmPmodule_target(
// CHECK: fir.store %[[N:.*]] to %{{.*}} <{tbaa = [#[[TARGET_T_TAG]]]}> : !fir.heap<i32>
// CHECK: %[[NEW:.*]] = fir.allocmem i32
// CHECK: fir.store %[[N]] to %[[NEW]] <{tbaa = [#[[TARGET_T_TAG]]]}> : !fir.heap<i32>
// CHECK: fir.load %{{.*}} <{tbaa = [#[[TARGET_T_TAG]]]}> : !fir.heap<i32>
func.func @_QMmPmodule_target(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<i32> {fir.bindc_name = "r"}) {
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMmEt) : !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %1 uniq_name("_QMmEt") fortran_attrs<allocatable, target> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFmodule_targetEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QMmFmodule_targetEr") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %5 = fir.load %3 : !fir.ref<i32>
  %6 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %8 = fir.convert %7 : (!fir.heap<i32>) -> i64
  %9 = arith.cmpi ne, %8, %c0_i64 : i64
  cf.cond_br %9, ^bb1, ^bb2
^bb1:
  fir.store %5 to %7 : !fir.heap<i32>
  cf.br ^bb3
^bb2:
  %10 = fir.allocmem i32
  fir.store %5 to %10 : !fir.heap<i32>
  %11 = fir.embox %10 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %11 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  cf.br ^bb3
^bb3:
  %12 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %13 = fir.box_addr %12 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %14 = fir.load %13 : !fir.heap<i32>
  fir.store %14 to %4 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @_QMmPlocal_var(
// CHECK: fir.store %[[N:.*]] to %{{.*}} <{tbaa = [#[[LOCAL_J_TAG]]]}> : !fir.heap<i32>
// CHECK: %[[NEW:.*]] = fir.allocmem i32
// CHECK: fir.store %[[N]] to %[[NEW]] <{tbaa = [#[[LOCAL_J_TAG]]]}> : !fir.heap<i32>
// CHECK: fir.load %{{.*}} <{tbaa = [#[[LOCAL_J_TAG]]]}> : !fir.heap<i32>
func.func @_QMmPlocal_var(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<i32> {fir.bindc_name = "r"}) {
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "j", uniq_name = "_QMmFlocal_varEj"}
  %2 = fir.zero_bits !fir.heap<i32>
  %3 = fir.embox %2 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %3 to %1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %4 = fir.declare %1 uniq_name("_QMmFlocal_varEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %5 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFlocal_varEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QMmFlocal_varEr") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %7 = fir.load %5 : !fir.ref<i32>
  %8 = fir.load %4 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %9 = fir.box_addr %8 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %10 = fir.convert %9 : (!fir.heap<i32>) -> i64
  %11 = arith.cmpi ne, %10, %c0_i64 : i64
  cf.cond_br %11, ^bb1, ^bb2
^bb1:
  fir.store %7 to %9 : !fir.heap<i32>
  cf.br ^bb3
^bb2:
  %12 = fir.allocmem i32
  fir.store %7 to %12 : !fir.heap<i32>
  %13 = fir.embox %12 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %13 to %4 : !fir.ref<!fir.box<!fir.heap<i32>>>
  cf.br ^bb3
^bb3:
  %14 = fir.load %4 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %15 = fir.box_addr %14 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %16 = fir.load %15 : !fir.heap<i32>
  fir.store %16 to %6 : !fir.ref<i32>
  return
}

fir.global @_QMmEt target : !fir.box<!fir.heap<i32>> {
  %0 = fir.zero_bits !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}
fir.global internal @_QMmFsave_varEj : !fir.box<!fir.heap<i32>> {
  %0 = fir.zero_bits !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}
}
