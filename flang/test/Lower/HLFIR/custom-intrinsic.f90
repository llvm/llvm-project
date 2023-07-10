! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

function max_simple(a, b)
  integer :: a, b, max_simple
  max_simple = max(a, b)
end function
! CHECK-LABEL: func.func @_QPmax_simple(
! CHECK-SAME:      %[[A_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}
! CHECK-SAME:      %[[B_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK-NEXT:    %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ARG]] {uniq_name = "_QFmax_simpleEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT:    %[[B_DECL:.*]]:2 = hlfir.declare %[[B_ARG]] {uniq_name = "_QFmax_simpleEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT:    %[[RES_ALLOC:.*]] = fir.alloca i32 {bindc_name = "max_simple", uniq_name = "_QFmax_simpleEmax_simple"}
! CHECK-NEXT:    %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOC]] {uniq_name = "_QFmax_simpleEmax_simple"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT:    %[[A_LD:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
! CHECK-NEXT:    %[[B_LD:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i32>
! CHECK-NEXT:    %[[A_GT_B:.*]] = arith.cmpi sgt, %[[A_LD]], %[[B_LD]] : i32
! CHECK-NEXT:    %[[SELECT:.*]] = arith.select %[[A_GT_B]], %[[A_LD]], %[[B_LD]] : i32
! CHECK-NEXT:    hlfir.assign %[[SELECT]] to %[[RES_DECL]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    %[[RES_LD:.*]] = fir.load %[[RES_DECL]]#1 : !fir.ref<i32>
! CHECK-NEXT:    return %[[RES_LD]] : i32
! CHECK-NEXT:  }

function max_dynamic_optional_scalar(a, b, c)
  integer :: a, b, max_dynamic_optional_scalar
  integer, optional :: c
  max_dynamic_optional_scalar = max(a, b, c)
end function
! CHECK-LABEL:   func.func @_QPmax_dynamic_optional_scalar(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                                              %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"},
! CHECK-SAME:                                              %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional}) -> i32 {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmax_dynamic_optional_scalarEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFmax_dynamic_optional_scalarEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFmax_dynamic_optional_scalarEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "max_dynamic_optional_scalar", uniq_name = "_QFmax_dynamic_optional_scalarEmax_dynamic_optional_scalar"}
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFmax_dynamic_optional_scalarEmax_dynamic_optional_scalar"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.is_present %[[VAL_5]]#0 : (!fir.ref<i32>) -> i1
! CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:           %[[VAL_13:.*]] = fir.if %[[VAL_10]] -> (i32) {
! CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_14]] : i32
! CHECK:             %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_12]], %[[VAL_14]] : i32
! CHECK:             fir.result %[[VAL_16]] : i32
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_12]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_17:.*]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<i32>
! CHECK:           return %[[VAL_18]] : i32
! CHECK:         }

function max_dynamic_optional_scalar2(a, b, c, d)
  integer :: a, b, max_dynamic_optional_scalar2
  integer, optional :: c, d
  max_dynamic_optional_scalar2 = max(a, b, c, d)
end function
! CHECK-LABEL:   func.func @_QPmax_dynamic_optional_scalar2(
! CHECK-SAME:                                               %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                                               %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"},
! CHECK-SAME:                                               %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional},
! CHECK-SAME:                                               %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "d", fir.optional}) -> i32 {
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmax_dynamic_optional_scalar2Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFmax_dynamic_optional_scalar2Eb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFmax_dynamic_optional_scalar2Ec"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFmax_dynamic_optional_scalar2Ed"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.alloca i32 {bindc_name = "max_dynamic_optional_scalar2", uniq_name = "_QFmax_dynamic_optional_scalar2Emax_dynamic_optional_scalar2"}
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFmax_dynamic_optional_scalar2Emax_dynamic_optional_scalar2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_12:.*]] = fir.is_present %[[VAL_6]]#0 : (!fir.ref<i32>) -> i1
! CHECK:           %[[VAL_13:.*]] = fir.is_present %[[VAL_7]]#0 : (!fir.ref<i32>) -> i1
! CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_16:.*]] = fir.if %[[VAL_12]] -> (i32) {
! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_17]] : i32
! CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_15]], %[[VAL_17]] : i32
! CHECK:             fir.result %[[VAL_19]] : i32
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_15]] : i32
! CHECK:           }
! CHECK:           %[[VAL_20:.*]] = fir.if %[[VAL_13]] -> (i32) {
! CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_23:.*]], %[[VAL_21]] : i32
! CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_22]], %[[VAL_23]], %[[VAL_21]] : i32
! CHECK:             fir.result %[[VAL_24]] : i32
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_25:.*]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_26:.*]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_9]]#1 : !fir.ref<i32>
! CHECK:           return %[[VAL_27]] : i32
! CHECK:         }

function min_simple(a, b)
  integer :: a, b, min_simple
  min_simple = min(a, b)
end function
! CHECK-LABEL:   func.func @_QPmin_simple(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                             %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}) -> i32 {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmin_simpleEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFmin_simpleEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "min_simple", uniq_name = "_QFmin_simpleEmin_simple"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFmin_simpleEmin_simple"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_7]] : i32
! CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_6]], %[[VAL_7]] : i32
! CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<i32>
! CHECK:           return %[[VAL_10]] : i32
! CHECK:         }

function min_dynamic_optional_scalar(a, b, c)
  integer :: a, b, min_dynamic_optional_scalar
  integer, optional :: c
  min_dynamic_optional_scalar = min(a, b, c)
end function
! CHECK-LABEL:   func.func @_QPmin_dynamic_optional_scalar(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                                              %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"},
! CHECK-SAME:                                              %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional}) -> i32 {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmin_dynamic_optional_scalarEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFmin_dynamic_optional_scalarEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFmin_dynamic_optional_scalarEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "min_dynamic_optional_scalar", uniq_name = "_QFmin_dynamic_optional_scalarEmin_dynamic_optional_scalar"}
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFmin_dynamic_optional_scalarEmin_dynamic_optional_scalar"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.is_present %[[VAL_5]]#0 : (!fir.ref<i32>) -> i1
! CHECK:           %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:           %[[VAL_13:.*]] = fir.if %[[VAL_10]] -> (i32) {
! CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_14]] : i32
! CHECK:             %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_12]], %[[VAL_14]] : i32
! CHECK:             fir.result %[[VAL_16]] : i32
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_12]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_17:.*]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<i32>
! CHECK:           return %[[VAL_18]] : i32
! CHECK:         }

function min_dynamic_optional_scalar2(a, b, c, d)
  integer :: a, b, min_dynamic_optional_scalar2
  integer, optional :: c, d
  min_dynamic_optional_scalar2 = min(a, b, c, d)
end function
! CHECK-LABEL:   func.func @_QPmin_dynamic_optional_scalar2(
! CHECK-SAME:                                               %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                                               %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"},
! CHECK-SAME:                                               %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional},
! CHECK-SAME:                                               %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "d", fir.optional}) -> i32 {
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmin_dynamic_optional_scalar2Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFmin_dynamic_optional_scalar2Eb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFmin_dynamic_optional_scalar2Ec"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFmin_dynamic_optional_scalar2Ed"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.alloca i32 {bindc_name = "min_dynamic_optional_scalar2", uniq_name = "_QFmin_dynamic_optional_scalar2Emin_dynamic_optional_scalar2"}
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFmin_dynamic_optional_scalar2Emin_dynamic_optional_scalar2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_12:.*]] = fir.is_present %[[VAL_6]]#0 : (!fir.ref<i32>) -> i1
! CHECK:           %[[VAL_13:.*]] = fir.is_present %[[VAL_7]]#0 : (!fir.ref<i32>) -> i1
! CHECK:           %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_16:.*]] = fir.if %[[VAL_12]] -> (i32) {
! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_17]] : i32
! CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_15]], %[[VAL_17]] : i32
! CHECK:             fir.result %[[VAL_19]] : i32
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_15]] : i32
! CHECK:           }
! CHECK:           %[[VAL_20:.*]] = fir.if %[[VAL_13]] -> (i32) {
! CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_23:.*]], %[[VAL_21]] : i32
! CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_22]], %[[VAL_23]], %[[VAL_21]] : i32
! CHECK:             fir.result %[[VAL_24]] : i32
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_25:.*]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_26:.*]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_9]]#1 : !fir.ref<i32>
! CHECK:           return %[[VAL_27]] : i32
! CHECK:         }

function associated_simple(pointer)
    integer, pointer :: pointer
    logical :: associated_simple
    associated_simple = associated(pointer)
end function
! CHECK-LABEL:   func.func @_QPassociated_simple(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "pointer"}) -> !fir.logical<4> {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.logical<4> {bindc_name = "associated_simple", uniq_name = "_QFassociated_simpleEassociated_simple"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFassociated_simpleEassociated_simple"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFassociated_simpleEpointer"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ptr<i32>) -> i64
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_2]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_2]]#1 : !fir.ref<!fir.logical<4>>
! CHECK:           return %[[VAL_10]] : !fir.logical<4>
! CHECK:         }

function associated_target(pointer, target)
    integer, pointer :: pointer
    integer, target :: target
    logical :: associated_target
    associated_target = associated(pointer, target)
end function
! CHECK-LABEL:   func.func @_QPassociated_target(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "pointer"},
! CHECK-SAME:                                    %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "target", fir.target}) -> !fir.logical<4> {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.logical<4> {bindc_name = "associated_target", uniq_name = "_QFassociated_targetEassociated_target"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFassociated_targetEassociated_target"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFassociated_targetEpointer"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFassociated_targetEtarget"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]]#1 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_4]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.ptr<i32>>) -> !fir.box<none>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_11]] to %[[VAL_3]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.logical<4>>
! CHECK:           return %[[VAL_12]] : !fir.logical<4>
! CHECK:         }

function associated_pointer(pointer, target)
    integer, pointer :: pointer
    integer, pointer :: target
    logical :: associated_pointer
    associated_pointer = associated(pointer, target)
end function
! CHECK-LABEL:   func.func @_QPassociated_pointer(
! CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "pointer"},
! CHECK-SAME:                                     %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "target"}) -> !fir.logical<4> {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.logical<4> {bindc_name = "associated_pointer", uniq_name = "_QFassociated_pointerEassociated_pointer"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFassociated_pointerEassociated_pointer"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFassociated_pointerEpointer"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFassociated_pointerEtarget"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_4]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.ptr<i32>>) -> !fir.box<none>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.ptr<i32>>) -> !fir.box<none>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_11]] to %[[VAL_3]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.logical<4>>
! CHECK:           return %[[VAL_12]] : !fir.logical<4>
! CHECK:         }

function associated_array(pointer, target)
    integer, pointer :: pointer(:)
    integer, pointer :: target(:)
    logical :: associated_array
    associated_array = associated(pointer, target)
end function
! CHECK-LABEL:   func.func @_QPassociated_array(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "pointer"},
! CHECK-SAME:                                   %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "target"}) -> !fir.logical<4> {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.logical<4> {bindc_name = "associated_array", uniq_name = "_QFassociated_arrayEassociated_array"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFassociated_arrayEassociated_array"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFassociated_arrayEpointer"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFassociated_arrayEtarget"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_4]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_11]] to %[[VAL_3]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.logical<4>>
! CHECK:           return %[[VAL_12]] : !fir.logical<4>
! CHECK:         }

function ishftc_simple(i, shift, size)
    integer :: ishftc_simple, i, shift, size
    ishftc_simple = ishftc(i, shift, size)
end function
! CHECK-LABEL:   func.func @_QPishftc_simple(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:                                %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "shift"},
! CHECK-SAME:                                %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "size"}) -> i32 {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFishftc_simpleEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "ishftc_simple", uniq_name = "_QFishftc_simpleEishftc_simple"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFishftc_simpleEishftc_simple"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFishftc_simpleEshift"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFishftc_simpleEsize"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = arith.constant 32 : i32
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_13:.*]] = arith.constant -1 : i32
! CHECK:           %[[VAL_14:.*]] = arith.constant 31 : i32
! CHECK:           %[[VAL_15:.*]] = arith.shrsi %[[VAL_9]], %[[VAL_14]] : i32
! CHECK:           %[[VAL_16:.*]] = arith.xori %[[VAL_9]], %[[VAL_15]] : i32
! CHECK:           %[[VAL_17:.*]] = arith.subi %[[VAL_16]], %[[VAL_15]] : i32
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_10]], %[[VAL_17]] : i32
! CHECK:           %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_12]] : i32
! CHECK:           %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_17]], %[[VAL_10]] : i32
! CHECK:           %[[VAL_21:.*]] = arith.ori %[[VAL_19]], %[[VAL_20]] : i1
! CHECK:           %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_12]] : i32
! CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_17]], %[[VAL_18]] : i32
! CHECK:           %[[VAL_24:.*]] = arith.select %[[VAL_22]], %[[VAL_18]], %[[VAL_17]] : i32
! CHECK:           %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_26:.*]] = arith.shrui %[[VAL_8]], %[[VAL_10]] : i32
! CHECK:           %[[VAL_27:.*]] = arith.shli %[[VAL_26]], %[[VAL_10]] : i32
! CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_25]], %[[VAL_27]], %[[VAL_12]] : i32
! CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_11]], %[[VAL_23]] : i32
! CHECK:           %[[VAL_30:.*]] = arith.shrui %[[VAL_13]], %[[VAL_29]] : i32
! CHECK:           %[[VAL_31:.*]] = arith.shrui %[[VAL_8]], %[[VAL_24]] : i32
! CHECK:           %[[VAL_32:.*]] = arith.andi %[[VAL_31]], %[[VAL_30]] : i32
! CHECK:           %[[VAL_33:.*]] = arith.subi %[[VAL_11]], %[[VAL_24]] : i32
! CHECK:           %[[VAL_34:.*]] = arith.shrui %[[VAL_13]], %[[VAL_33]] : i32
! CHECK:           %[[VAL_35:.*]] = arith.andi %[[VAL_8]], %[[VAL_34]] : i32
! CHECK:           %[[VAL_36:.*]] = arith.shli %[[VAL_35]], %[[VAL_23]] : i32
! CHECK:           %[[VAL_37:.*]] = arith.ori %[[VAL_28]], %[[VAL_32]] : i32
! CHECK:           %[[VAL_38:.*]] = arith.ori %[[VAL_37]], %[[VAL_36]] : i32
! CHECK:           %[[VAL_39:.*]] = arith.select %[[VAL_21]], %[[VAL_8]], %[[VAL_38]] : i32
! CHECK:           hlfir.assign %[[VAL_39]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<i32>
! CHECK:           return %[[VAL_40]] : i32
! CHECK:         }

function ishftc_dynamically_optional_scalar(i, shift, size)
    integer :: ishftc_dynamically_optional_scalar, i, shift
    integer, optional :: size
    ishftc_dynamically_optional_scalar = ishftc(i, shift, size)
end function
! CHECK-LABEL:   func.func @_QPishftc_dynamically_optional_scalar(
! CHECK-SAME:                                                     %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:                                                     %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "shift"},
! CHECK-SAME:                                                     %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "size", fir.optional}) -> i32 {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFishftc_dynamically_optional_scalarEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "ishftc_dynamically_optional_scalar", uniq_name = "_QFishftc_dynamically_optional_scalarEishftc_dynamically_optional_scalar"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFishftc_dynamically_optional_scalarEishftc_dynamically_optional_scalar"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFishftc_dynamically_optional_scalarEshift"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFishftc_dynamically_optional_scalarEsize"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.is_present %[[VAL_7]]#0 : (!fir.ref<i32>) -> i1
! CHECK:           %[[VAL_11:.*]] = fir.if %[[VAL_10]] -> (i32) {
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:             fir.result %[[VAL_12]] : i32
! CHECK:           } else {
! CHECK:             %[[VAL_13:.*]] = arith.constant 32 : i32
! CHECK:             fir.result %[[VAL_13]] : i32
! CHECK:           }
! CHECK:           %[[VAL_14:.*]] = arith.constant 32 : i32
! CHECK:           %[[VAL_15:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_16:.*]] = arith.constant -1 : i32
! CHECK:           %[[VAL_17:.*]] = arith.constant 31 : i32
! CHECK:           %[[VAL_18:.*]] = arith.shrsi %[[VAL_9]], %[[VAL_17]] : i32
! CHECK:           %[[VAL_19:.*]] = arith.xori %[[VAL_9]], %[[VAL_18]] : i32
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : i32
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_22:.*]], %[[VAL_20]] : i32
! CHECK:           %[[VAL_23:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_15]] : i32
! CHECK:           %[[VAL_24:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_22]] : i32
! CHECK:           %[[VAL_25:.*]] = arith.ori %[[VAL_23]], %[[VAL_24]] : i1
! CHECK:           %[[VAL_26:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_15]] : i32
! CHECK:           %[[VAL_27:.*]] = arith.select %[[VAL_26]], %[[VAL_20]], %[[VAL_21]] : i32
! CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_26]], %[[VAL_21]], %[[VAL_20]] : i32
! CHECK:           %[[VAL_29:.*]] = arith.cmpi ne, %[[VAL_22]], %[[VAL_14]] : i32
! CHECK:           %[[VAL_30:.*]] = arith.shrui %[[VAL_8]], %[[VAL_22]] : i32
! CHECK:           %[[VAL_31:.*]] = arith.shli %[[VAL_30]], %[[VAL_22]] : i32
! CHECK:           %[[VAL_32:.*]] = arith.select %[[VAL_29]], %[[VAL_31]], %[[VAL_15]] : i32
! CHECK:           %[[VAL_33:.*]] = arith.subi %[[VAL_14]], %[[VAL_27]] : i32
! CHECK:           %[[VAL_34:.*]] = arith.shrui %[[VAL_16]], %[[VAL_33]] : i32
! CHECK:           %[[VAL_35:.*]] = arith.shrui %[[VAL_8]], %[[VAL_28]] : i32
! CHECK:           %[[VAL_36:.*]] = arith.andi %[[VAL_35]], %[[VAL_34]] : i32
! CHECK:           %[[VAL_37:.*]] = arith.subi %[[VAL_14]], %[[VAL_28]] : i32
! CHECK:           %[[VAL_38:.*]] = arith.shrui %[[VAL_16]], %[[VAL_37]] : i32
! CHECK:           %[[VAL_39:.*]] = arith.andi %[[VAL_8]], %[[VAL_38]] : i32
! CHECK:           %[[VAL_40:.*]] = arith.shli %[[VAL_39]], %[[VAL_27]] : i32
! CHECK:           %[[VAL_41:.*]] = arith.ori %[[VAL_32]], %[[VAL_36]] : i32
! CHECK:           %[[VAL_42:.*]] = arith.ori %[[VAL_41]], %[[VAL_40]] : i32
! CHECK:           %[[VAL_43:.*]] = arith.select %[[VAL_25]], %[[VAL_8]], %[[VAL_42]] : i32
! CHECK:           hlfir.assign %[[VAL_43]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_44:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<i32>
! CHECK:           return %[[VAL_44]] : i32
! CHECK:         }