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
//   subroutine escaped(n)
//     integer :: n
//     integer, allocatable :: j
//     j = n
//   end subroutine
//   subroutine nl2(n, r)
//     integer :: n, r
//     integer :: a, b
//     namelist /g/ a, b
//     a = n
//     b = n + 1
//     write (*, g)
//     r = a + b
//   end subroutine
// end module
//
// In escaped, the address of the new buffer is also stored into a tuple. The
// analysis cannot tell where the address goes, so the store into the buffer
// gets no tag.
//
// In nl2, the address of the item list of the NAMELIST group is stored into
// the group, through which the runtime reads the list. So the store into the
// list gets no tag either, while the accesses of a and b keep their tags.

// RUN: fir-opt --fir-add-alias-tags %s | FileCheck %s

// CHECK-DAG: #[[DUMMY_J:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QMmFdummy_varEj", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[DIRECT_J:.+]] = #llvm.tbaa_type_desc<id = "direct data/_QMmFsave_varEj", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[TARGET_T:.+]] = #llvm.tbaa_type_desc<id = "target data/_QMmEt", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[LOCAL_J:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QMmFlocal_varEj", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[DUMMY_J_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[DUMMY_J]], access_type = #[[DUMMY_J]], offset = 0>
// CHECK-DAG: #[[DIRECT_J_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[DIRECT_J]], access_type = #[[DIRECT_J]], offset = 0>
// CHECK-DAG: #[[TARGET_T_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[TARGET_T]], access_type = #[[TARGET_T]], offset = 0>
// CHECK-DAG: #[[LOCAL_J_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[LOCAL_J]], access_type = #[[LOCAL_J]], offset = 0>
// CHECK-DAG: #[[NL2_A:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QMmFnl2Ea", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[NL2_B:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QMmFnl2Eb", members = {<#{{.+}}, 0>}>
// CHECK-DAG: #[[NL2_A_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[NL2_A]], access_type = #[[NL2_A]], offset = 0>
// CHECK-DAG: #[[NL2_B_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[NL2_B]], access_type = #[[NL2_B]], offset = 0>

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

// CHECK-LABEL: func.func @_QMmPescaped(
// CHECK: %[[NEW:.*]] = fir.allocmem i32
// CHECK: fir.store %{{.*}} to %[[NEW]] : !fir.heap<i32>
func.func @_QMmPescaped(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "j", uniq_name = "_QMmFescapedEj"}
  %2 = fir.declare %1 uniq_name("_QMmFescapedEj") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFescapedEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  %5 = fir.allocmem i32
  fir.store %4 to %5 : !fir.heap<i32>
  %6 = fir.embox %5 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %6 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %7 = fir.alloca tuple<!fir.heap<i32>>
  %8 = fir.undefined tuple<!fir.heap<i32>>
  %9 = fir.insert_value %8, %5, [0 : index] : (tuple<!fir.heap<i32>>, !fir.heap<i32>) -> tuple<!fir.heap<i32>>
  fir.store %9 to %7 : !fir.ref<tuple<!fir.heap<i32>>>
  return
}

// CHECK-LABEL: func.func @_QMmPnl2(
// CHECK: %[[A:.*]] = fir.declare %{{.*}} uniq_name("_QMmFnl2Ea")
// CHECK: %[[B:.*]] = fir.declare %{{.*}} uniq_name("_QMmFnl2Eb")
// CHECK: fir.store %{{.*}} to %[[A]] <{tbaa = [#[[NL2_A_TAG]]]}> : !fir.ref<i32>
// CHECK: fir.store %{{.*}} to %[[B]] <{tbaa = [#[[NL2_B_TAG]]]}> : !fir.ref<i32>
// CHECK: %[[LIST:.*]] = fir.alloca !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
// CHECK: fir.store %{{.*}} to %[[LIST]] : !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>
// CHECK: fir.call @_FortranAioOutputNamelist(
// CHECK: fir.load %[[A]] <{tbaa = [#[[NL2_A_TAG]]]}> : !fir.ref<i32>
// CHECK: fir.load %[[B]] <{tbaa = [#[[NL2_B_TAG]]]}> : !fir.ref<i32>
func.func @_QMmPnl2(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<i32> {fir.bindc_name = "r"}) {
  %c2_i64 = arith.constant 2 : i64
  %c9_i32 = arith.constant 9 : i32
  %c6_i32 = arith.constant 6 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca !fir.box<!fir.ptr<i32>>
  %1 = fir.alloca !fir.box<!fir.ptr<i32>>
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.alloca i32 <{bindc_name = "a", uniq_name = "_QMmFnl2Ea"}>
  %4 = fir.declare %3 uniq_name("_QMmFnl2Ea") : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.alloca i32 <{bindc_name = "b", uniq_name = "_QMmFnl2Eb"}>
  %6 = fir.declare %5 uniq_name("_QMmFnl2Eb") : (!fir.ref<i32>) -> !fir.ref<i32>
  %7 = fir.declare %arg0 dummy_scope %2 arg 1 uniq_name("_QMmFnl2En") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %8 = fir.declare %arg1 dummy_scope %2 arg 2 uniq_name("_QMmFnl2Er") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %9 = fir.load %7 : !fir.ref<i32>
  fir.store %9 to %4 : !fir.ref<i32>
  %10 = fir.load %7 : !fir.ref<i32>
  %11 = arith.addi %10, %c1_i32 : i32
  fir.store %11 to %6 : !fir.ref<i32>
  %12 = fir.address_of(@_QQclX6E6C326D2E66393000) : !fir.ref<!fir.char<1,9>>
  %13 = fir.convert %12 : (!fir.ref<!fir.char<1,9>>) -> !fir.ref<i8>
  %14 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %13, %c9_i32) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  %15 = fir.alloca !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  %16 = fir.undefined !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  %17 = fir.address_of(@_QQclX6100) : !fir.ref<!fir.char<1,2>>
  %18 = fir.convert %17 : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
  %19 = fir.insert_value %16, %18, [0 : index, 0 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  %20 = fir.embox %4 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  fir.store %20 to %1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %21 = fir.convert %1 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
  %22 = fir.insert_value %19, %21, [0 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  %23 = fir.address_of(@_QQclX6200) : !fir.ref<!fir.char<1,2>>
  %24 = fir.convert %23 : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
  %25 = fir.insert_value %22, %24, [1 : index, 0 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  %26 = fir.embox %6 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  fir.store %26 to %0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %27 = fir.convert %0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
  %28 = fir.insert_value %25, %27, [1 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  fir.store %28 to %15 : !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>
  %29 = fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  %30 = fir.undefined tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  %31 = fir.address_of(@_QQclX6700) : !fir.ref<!fir.char<1,2>>
  %32 = fir.convert %31 : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
  %33 = fir.insert_value %30, %32, [0 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<i8>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  %34 = fir.insert_value %33, %c2_i64, [1 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, i64) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  %35 = fir.insert_value %34, %15, [2 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  %36 = fir.address_of(@_QQdefault.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>>
  %37 = fir.convert %36 : (!fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>>) -> !fir.ref<none>
  %38 = fir.insert_value %35, %37, [3 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<none>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  fir.store %38 to %29 : !fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>
  %39 = fir.convert %29 : (!fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>) -> !fir.ref<tuple<>>
  %40 = fir.call @_FortranAioOutputNamelist(%14, %39) : (!fir.ref<i8>, !fir.ref<tuple<>>) -> i1
  %41 = fir.call @_FortranAioEndIoStatement(%14) : (!fir.ref<i8>) -> i32
  %42 = fir.load %4 : !fir.ref<i32>
  %43 = fir.load %6 : !fir.ref<i32>
  %44 = arith.addi %42, %43 : i32
  fir.store %44 to %8 : !fir.ref<i32>
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
func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX6E6C326D2E66393000 constant : !fir.char<1,9> {
  %0 = fir.string_lit "nl2m.f90\00"(9) : !fir.char<1,9>
  fir.has_value %0 : !fir.char<1,9>
}
func.func private @_FortranAioOutputNamelist(!fir.ref<i8>, !fir.ref<tuple<>>) -> i1 attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX6700 constant : !fir.char<1,2> {
  %0 = fir.string_lit "g\00"(2) : !fir.char<1,2>
  fir.has_value %0 : !fir.char<1,2>
}
fir.global linkonce @_QQclX6100 constant : !fir.char<1,2> {
  %0 = fir.string_lit "a\00"(2) : !fir.char<1,2>
  fir.has_value %0 : !fir.char<1,2>
}
fir.global linkonce @_QQclX6200 constant : !fir.char<1,2> {
  %0 = fir.string_lit "b\00"(2) : !fir.char<1,2>
  fir.has_value %0 : !fir.char<1,2>
}
fir.global linkonce @_QQdefault.nonTbpDefinedIoTable constant : tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1> {
  %true = arith.constant true
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.undefined tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>
  %1 = fir.insert_value %0, %c0_i64, [0 : index] : (tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>, i64) -> tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>
  %2 = fir.zero_bits !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>
  %3 = fir.insert_value %1, %2, [1 : index] : (tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>) -> tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>
  %4 = fir.insert_value %3, %true, [2 : index] : (tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>, i1) -> tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>
  fir.has_value %4 : tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i8>>>, i1>
}
func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}
}
