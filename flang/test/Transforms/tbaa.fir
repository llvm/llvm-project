// RUN: fir-opt --split-input-file --fir-add-alias-tags %s | FileCheck %s

// subroutine oneArg(a)
//   integer :: a(:)
//   a(1) = a(2)
// end subroutine
  func.func @_QPonearg(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFoneargEa"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %1 = fir.rebox %0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %2 = fir.array_coor %1 %c2 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %3 = fir.load %2 : !fir.ref<i32>
    %4 = fir.array_coor %1 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %3 to %4 : !fir.ref<i32>
    return
  }

// CHECK: #[[ONE_ARG_ROOT:.+]] = #ptr.tbaa_root<id = "Flang function root _QPonearg">
// CHECK: #[[ONE_ARG_ANY_ACCESS:.+]] = #ptr.tbaa_type_desc<id = "any access", members = {<#[[ONE_ARG_ROOT]], 0>}>
// CHECK: #[[ONE_ARG_ANY_DATA:.+]] = #ptr.tbaa_type_desc<id = "any data access", members = {<#[[ONE_ARG_ANY_ACCESS]], 0>}>
// CHECK: #[[ONE_ARG_ANY_ARG:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data", members = {<#[[ONE_ARG_ANY_DATA]], 0>}>
// CHECK: #[[ONE_ARG_A:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data/_QFoneargEa", members = {<#[[ONE_ARG_ANY_ARG]], 0>}>
// CHECK: #[[ONE_ARG_A_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[ONE_ARG_A]], access_type = #[[ONE_ARG_A]], offset = 0>

// CHECK-LABEL:   func.func @_QPonearg(
// CHECK-SAME:                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFoneargEa"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_4:.*]] = fir.rebox %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_5:.*]] = fir.array_coor %[[VAL_4]] %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]] {tbaa = [#[[ONE_ARG_A_TAG]]]} : !fir.ref<i32>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_4]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           fir.store %[[VAL_6]] to %[[VAL_7]] {tbaa = [#[[ONE_ARG_A_TAG]]]} : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// -----

// subroutine twoArg(a, b)
//   integer :: a(:), b(:)
//   a(1) = b(1)
// end subroutine
  func.func @_QPtwoarg(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) {
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFtwoargEa"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %1 = fir.rebox %0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %2 = fir.declare %arg1 {uniq_name = "_QFtwoargEb"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %3 = fir.rebox %2 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %4 = fir.array_coor %3 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %5 = fir.load %4 : !fir.ref<i32>
    %6 = fir.array_coor %1 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %5 to %6 : !fir.ref<i32>
    return
  }

// CHECK: #[[TWO_ARG_ROOT:.+]] = #ptr.tbaa_root<id = "Flang function root _QPtwoarg">
// CHECK: #[[TWO_ARG_ANY_ACCESS:.+]] = #ptr.tbaa_type_desc<id = "any access", members = {<#[[TWO_ARG_ROOT]], 0>}>
// CHECK: #[[TWO_ARG_ANY_DATA:.+]] = #ptr.tbaa_type_desc<id = "any data access", members = {<#[[TWO_ARG_ANY_ACCESS]], 0>}>
// CHECK: #[[TWO_ARG_ANY_ARG:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data", members = {<#[[TWO_ARG_ANY_DATA]], 0>}>
// CHECK: #[[TWO_ARG_B:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data/_QFtwoargEb", members = {<#[[TWO_ARG_ANY_ARG]], 0>}>
// CHECK: #[[TWO_ARG_A:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data/_QFtwoargEa", members = {<#[[TWO_ARG_ANY_ARG]], 0>}>
// CHECK: #[[TWO_ARG_B_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[TWO_ARG_B]], access_type = #[[TWO_ARG_B]], offset = 0>
// CHECK: #[[TWO_ARG_A_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[TWO_ARG_A]], access_type = #[[TWO_ARG_A]], offset = 0>

// CHECK-LABEL:   func.func @_QPtwoarg(
// CHECK-SAME:                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:                         %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFtwoargEa"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_4:.*]] = fir.rebox %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFtwoargEb"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_6]] %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] {tbaa = [#[[TWO_ARG_B_TAG]]]} : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.array_coor %[[VAL_4]] %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           fir.store %[[VAL_8]] to %[[VAL_9]] {tbaa = [#[[TWO_ARG_A_TAG]]]} : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// -----

// subroutine targetArg(a, b)
//   integer, target :: a(:)
//   integer :: b(:)
//   a(1) = b(1)
// end subroutine
  func.func @_QPtargetarg(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a", fir.target}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) {
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtargetargEa"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %1 = fir.rebox %0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %2 = fir.declare %arg1 {uniq_name = "_QFtargetargEb"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %3 = fir.rebox %2 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %4 = fir.array_coor %3 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %5 = fir.load %4 : !fir.ref<i32>
    %6 = fir.array_coor %1 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %5 to %6 : !fir.ref<i32>
    return
  }

// CHECK: #[[TARGET_ROOT:.+]] = #ptr.tbaa_root<id = "Flang function root _QPtargetarg">
// CHECK: #[[TARGET_ANY_ACCESS:.+]] = #ptr.tbaa_type_desc<id = "any access", members = {<#[[TARGET_ROOT]], 0>}>
// CHECK: #[[TARGET_ANY_DATA:.+]] = #ptr.tbaa_type_desc<id = "any data access", members = {<#[[TARGET_ANY_ACCESS]], 0>}>
// CHECK: #[[TARGET_ANY_ARG:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data", members = {<#[[TARGET_ANY_DATA]], 0>}>
// CHECK: #[[TARGET_B:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data/_QFtargetargEb", members = {<#[[TARGET_ANY_ARG]], 0>}>
// CHECK: #[[TARGET_B_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[TARGET_B]], access_type = #[[TARGET_B]], offset = 0>
// No entry for "dummy arg data/a" because that pointer should get "any data access" becase it has the TARGET attribute

// CHECK-LABEL:   func.func @_QPtargetarg(
// CHECK-SAME:                            %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a", fir.target},
// CHECK-SAME:                            %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {fortran_attrs = #{{.*}}<target>, uniq_name = "_QFtargetargEa"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_4:.*]] = fir.rebox %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFtargetargEb"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_6]] %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] {tbaa = [#[[TARGET_B_TAG]]]} : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.array_coor %[[VAL_4]] %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// "any data access" tag is added by TBAABuilder during CodeGen
// CHECK:           fir.store %[[VAL_8]] to %[[VAL_9]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// -----

// subroutine pointerArg(a, b)
//   integer, pointer :: a(:)
//   integer :: b(:)
//   a(1) = b(1)
// end subroutine
  func.func @_QPpointerarg(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFpointerargEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %1 = fir.declare %arg1 {uniq_name = "_QFpointerargEb"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %2 = fir.rebox %1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
    %3 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %4 = fir.load %3 : !fir.ref<i32>
    %5 = fir.load %0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %6:3 = fir.box_dims %5, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %7 = fir.shift %6#0 : (index) -> !fir.shift<1>
    %8 = fir.array_coor %5(%7) %c1 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, index) -> !fir.ref<i32>
    fir.store %4 to %8 : !fir.ref<i32>
    return
  }

// CHECK: #[[POINTER_ROOT:.+]] = #ptr.tbaa_root<id = "Flang function root _QPpointerarg">
// CHECK: #[[POINTER_ANY_ACCESS:.+]] = #ptr.tbaa_type_desc<id = "any access", members = {<#[[POINTER_ROOT]], 0>}>
// CHECK: #[[POINTER_ANY_DATA:.+]] = #ptr.tbaa_type_desc<id = "any data access", members = {<#[[POINTER_ANY_ACCESS]], 0>}>
// CHECK: #[[POINTER_ANY_ARG:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data", members = {<#[[POINTER_ANY_DATA]], 0>}>
// CHECK: #[[POINTER_B:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data/_QFpointerargEb", members = {<#[[POINTER_ANY_ARG]], 0>}>
// CHECK: #[[POINTER_B_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[POINTER_B]], access_type = #[[POINTER_B]], offset = 0>
// No entry for "dummy arg data/a" because that pointer should get "any data access" becase it has the POINTER attribute

// CHECK-LABEL:   func.func @_QPpointerarg(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "a"},
// CHECK-SAME:                             %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_0]] {fortran_attrs = #{{.*}}<pointer>, uniq_name = "_QFpointerargEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFpointerargEb"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_6]] %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] {tbaa = [#[[POINTER_B_TAG]]]} : !fir.ref<i32>
// "any descriptor access" tag is added by TBAABuilder during CodeGen
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_11:.*]] = fir.shift %[[VAL_10]]#0 : (index) -> !fir.shift<1>
// CHECK:           %[[VAL_12:.*]] = fir.array_coor %[[VAL_9]](%[[VAL_11]]) %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, index) -> !fir.ref<i32>
// "any data access" tag is added by TBAABuilder during CodeGen
// CHECK:           fir.store %[[VAL_8]] to %[[VAL_12]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
