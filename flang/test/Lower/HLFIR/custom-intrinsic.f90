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

function max_array(a, b)
   integer, dimension(42) :: a, b, max_array
   max_array = max(a, b)
end function
! CHECK-LABEL:   func.func @_QPmax_array(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.array<42xi32>> {fir.bindc_name = "b"}) -> !fir.array<42xi32> {
! CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "_QFmax_arrayEa"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_6]]) {uniq_name = "_QFmax_arrayEb"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.array<42xi32> {bindc_name = "max_array", uniq_name = "_QFmax_arrayEmax_array"}
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) {uniq_name = "_QFmax_arrayEmax_array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_12:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
! CHECK:           ^bb0(%[[VAL_13:.*]]: index):
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_17]] : i32
! CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_15]], %[[VAL_17]] : i32
! CHECK:             hlfir.yield_element %[[VAL_19]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_20:.*]] to %[[VAL_11]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
! CHECK:           hlfir.destroy %[[VAL_20]] : !hlfir.expr<42xi32>
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_11]]#1 : !fir.ref<!fir.array<42xi32>>
! CHECK:           return %[[VAL_21]] : !fir.array<42xi32>
! CHECK:         }

function max_dynamic_optional_array(a, b, c)
   integer :: a, b(10), max_dynamic_optional_array(10)
   integer, optional :: c(10)
   max_dynamic_optional_array = max(a, b, c)
end function
! CHECK-LABEL:   func.func @_QPmax_dynamic_optional_array(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                                             %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "b"},
! CHECK-SAME:                                             %[[VAL_2:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "c", fir.optional}) -> !fir.array<10xi32> {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmax_dynamic_optional_arrayEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_5]]) {uniq_name = "_QFmax_dynamic_optional_arrayEb"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_8]]) {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFmax_dynamic_optional_arrayEc"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_11:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "max_dynamic_optional_array", uniq_name = "_QFmax_dynamic_optional_arrayEmax_dynamic_optional_array"}
! CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_12]]) {uniq_name = "_QFmax_dynamic_optional_arrayEmax_dynamic_optional_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_14:.*]] = fir.is_present %[[VAL_9]]#0 : (!fir.ref<!fir.array<10xi32>>) -> i1
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_17:.*]]: index):
! CHECK:             %[[VAL_18:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK:             %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_19]] : i32
! CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_15]], %[[VAL_19]] : i32
! CHECK:             %[[VAL_22:.*]] = fir.if %[[VAL_14]] -> (i32) {
! CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:               %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_21]], %[[VAL_24]] : i32
! CHECK:               %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_21]], %[[VAL_24]] : i32
! CHECK:               fir.result %[[VAL_26]] : i32
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_21]] : i32
! CHECK:             }
! CHECK:             hlfir.yield_element %[[VAL_27:.*]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_28:.*]] to %[[VAL_13]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_28]] : !hlfir.expr<10xi32>
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_13]]#1 : !fir.ref<!fir.array<10xi32>>
! CHECK:           return %[[VAL_29]] : !fir.array<10xi32>
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

function min_array(a, b)
   integer, dimension(42) :: a, b, min_array
   min_array = min(a, b)
end function
! CHECK-LABEL:   func.func @_QPmin_array(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.array<42xi32>> {fir.bindc_name = "b"}) -> !fir.array<42xi32> {
! CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "_QFmin_arrayEa"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_6]]) {uniq_name = "_QFmin_arrayEb"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.array<42xi32> {bindc_name = "min_array", uniq_name = "_QFmin_arrayEmin_array"}
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) {uniq_name = "_QFmin_arrayEmin_array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_12:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
! CHECK:           ^bb0(%[[VAL_13:.*]]: index):
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_17]] : i32
! CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_15]], %[[VAL_17]] : i32
! CHECK:             hlfir.yield_element %[[VAL_19]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_20:.*]] to %[[VAL_11]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
! CHECK:           hlfir.destroy %[[VAL_20]] : !hlfir.expr<42xi32>
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_11]]#1 : !fir.ref<!fir.array<42xi32>>
! CHECK:           return %[[VAL_21]] : !fir.array<42xi32>
! CHECK:         }

function min_dynamic_optional_array(a, b, c)
   integer :: a, b(10), min_dynamic_optional_array(10)
   integer, optional :: c(10)
   min_dynamic_optional_array = min(a, b, c)
end function
! CHECK-LABEL:   func.func @_QPmin_dynamic_optional_array(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                                             %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "b"},
! CHECK-SAME:                                             %[[VAL_2:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "c", fir.optional}) -> !fir.array<10xi32> {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmin_dynamic_optional_arrayEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_5]]) {uniq_name = "_QFmin_dynamic_optional_arrayEb"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_8]]) {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFmin_dynamic_optional_arrayEc"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_11:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "min_dynamic_optional_array", uniq_name = "_QFmin_dynamic_optional_arrayEmin_dynamic_optional_array"}
! CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_12]]) {uniq_name = "_QFmin_dynamic_optional_arrayEmin_dynamic_optional_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_14:.*]] = fir.is_present %[[VAL_9]]#0 : (!fir.ref<!fir.array<10xi32>>) -> i1
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<10xi32> {
! CHECK:           ^bb0(%[[VAL_17:.*]]: index):
! CHECK:             %[[VAL_18:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK:             %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_19]] : i32
! CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_15]], %[[VAL_19]] : i32
! CHECK:             %[[VAL_22:.*]] = fir.if %[[VAL_14]] -> (i32) {
! CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:               %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_24]] : i32
! CHECK:               %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_21]], %[[VAL_24]] : i32
! CHECK:               fir.result %[[VAL_26]] : i32
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_21]] : i32
! CHECK:             }
! CHECK:             hlfir.yield_element %[[VAL_27:.*]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_28:.*]] to %[[VAL_13]]#0 : !hlfir.expr<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.destroy %[[VAL_28]] : !hlfir.expr<10xi32>
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_13]]#1 : !fir.ref<!fir.array<10xi32>>
! CHECK:           return %[[VAL_29]] : !fir.array<10xi32>
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

function ishftc_array(i, shift, size)
   integer, dimension(42) :: ishftc_array, i, shift, size
   ishftc_array = ishftc(i, shift, size)
end function
! CHECK-LABEL:   func.func @_QPishftc_array(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                               %[[VAL_1:.*]]: !fir.ref<!fir.array<42xi32>> {fir.bindc_name = "shift"},
! CHECK-SAME:                               %[[VAL_2:.*]]: !fir.ref<!fir.array<42xi32>> {fir.bindc_name = "size"}) -> !fir.array<42xi32> {
! CHECK:           %[[VAL_3:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "_QFishftc_arrayEi"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.array<42xi32> {bindc_name = "ishftc_array", uniq_name = "_QFishftc_arrayEishftc_array"}
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) {uniq_name = "_QFishftc_arrayEishftc_array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_11]]) {uniq_name = "_QFishftc_arrayEshift"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_13:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_14]]) {uniq_name = "_QFishftc_arrayEsize"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_16:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
! CHECK:           ^bb0(%[[VAL_17:.*]]: index):
! CHECK:             %[[VAL_18:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK:             %[[VAL_20:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i32>
! CHECK:             %[[VAL_22:.*]] = hlfir.designate %[[VAL_15]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<i32>
! CHECK:             %[[VAL_24:.*]] = arith.constant 32 : i32
! CHECK:             %[[VAL_25:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_26:.*]] = arith.constant -1 : i32
! CHECK:             %[[VAL_27:.*]] = arith.constant 31 : i32
! CHECK:             %[[VAL_28:.*]] = arith.shrsi %[[VAL_21]], %[[VAL_27]] : i32
! CHECK:             %[[VAL_29:.*]] = arith.xori %[[VAL_21]], %[[VAL_28]] : i32
! CHECK:             %[[VAL_30:.*]] = arith.subi %[[VAL_29]], %[[VAL_28]] : i32
! CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_23]], %[[VAL_30]] : i32
! CHECK:             %[[VAL_32:.*]] = arith.cmpi eq, %[[VAL_21]], %[[VAL_25]] : i32
! CHECK:             %[[VAL_33:.*]] = arith.cmpi eq, %[[VAL_30]], %[[VAL_23]] : i32
! CHECK:             %[[VAL_34:.*]] = arith.ori %[[VAL_32]], %[[VAL_33]] : i1
! CHECK:             %[[VAL_35:.*]] = arith.cmpi sgt, %[[VAL_21]], %[[VAL_25]] : i32
! CHECK:             %[[VAL_36:.*]] = arith.select %[[VAL_35]], %[[VAL_30]], %[[VAL_31]] : i32
! CHECK:             %[[VAL_37:.*]] = arith.select %[[VAL_35]], %[[VAL_31]], %[[VAL_30]] : i32
! CHECK:             %[[VAL_38:.*]] = arith.cmpi ne, %[[VAL_23]], %[[VAL_24]] : i32
! CHECK:             %[[VAL_39:.*]] = arith.shrui %[[VAL_19]], %[[VAL_23]] : i32
! CHECK:             %[[VAL_40:.*]] = arith.shli %[[VAL_39]], %[[VAL_23]] : i32
! CHECK:             %[[VAL_41:.*]] = arith.select %[[VAL_38]], %[[VAL_40]], %[[VAL_25]] : i32
! CHECK:             %[[VAL_42:.*]] = arith.subi %[[VAL_24]], %[[VAL_36]] : i32
! CHECK:             %[[VAL_43:.*]] = arith.shrui %[[VAL_26]], %[[VAL_42]] : i32
! CHECK:             %[[VAL_44:.*]] = arith.shrui %[[VAL_19]], %[[VAL_37]] : i32
! CHECK:             %[[VAL_45:.*]] = arith.andi %[[VAL_44]], %[[VAL_43]] : i32
! CHECK:             %[[VAL_46:.*]] = arith.subi %[[VAL_24]], %[[VAL_37]] : i32
! CHECK:             %[[VAL_47:.*]] = arith.shrui %[[VAL_26]], %[[VAL_46]] : i32
! CHECK:             %[[VAL_48:.*]] = arith.andi %[[VAL_19]], %[[VAL_47]] : i32
! CHECK:             %[[VAL_49:.*]] = arith.shli %[[VAL_48]], %[[VAL_36]] : i32
! CHECK:             %[[VAL_50:.*]] = arith.ori %[[VAL_41]], %[[VAL_45]] : i32
! CHECK:             %[[VAL_51:.*]] = arith.ori %[[VAL_50]], %[[VAL_49]] : i32
! CHECK:             %[[VAL_52:.*]] = arith.select %[[VAL_34]], %[[VAL_19]], %[[VAL_51]] : i32
! CHECK:             hlfir.yield_element %[[VAL_52]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_53:.*]] to %[[VAL_9]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
! CHECK:           hlfir.destroy %[[VAL_53]] : !hlfir.expr<42xi32>
! CHECK:           %[[VAL_54:.*]] = fir.load %[[VAL_9]]#1 : !fir.ref<!fir.array<42xi32>>
! CHECK:           return %[[VAL_54]] : !fir.array<42xi32>
! CHECK:         }

function ishftc_dynamically_optional_array(i, shift, size)
   integer :: ishftc_dynamically_optional_array(42), i(42), shift
   integer, optional :: size
   ishftc_dynamically_optional_array = ishftc(i, shift, size)
end function
! CHECK-LABEL:   func.func @_QPishftc_dynamically_optional_array(
! CHECK-SAME:                                                    %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                                                    %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "shift"},
! CHECK-SAME:                                                    %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "size", fir.optional}) -> !fir.array<42xi32> {
! CHECK:           %[[VAL_3:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "_QFishftc_dynamically_optional_arrayEi"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 42 : index
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.array<42xi32> {bindc_name = "ishftc_dynamically_optional_array", uniq_name = "_QFishftc_dynamically_optional_arrayEishftc_dynamically_optional_array"}
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) {uniq_name = "_QFishftc_dynamically_optional_arrayEishftc_dynamically_optional_array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFishftc_dynamically_optional_arrayEshift"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFishftc_dynamically_optional_arrayEsize"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_12:.*]] = fir.is_present %[[VAL_11]]#0 : (!fir.ref<i32>) -> i1
! CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
! CHECK:           ^bb0(%[[VAL_15:.*]]: index):
! CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_15]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = fir.if %[[VAL_12]] -> (i32) {
! CHECK:               %[[VAL_19:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<i32>
! CHECK:               fir.result %[[VAL_19]] : i32
! CHECK:             } else {
! CHECK:               %[[VAL_20:.*]] = arith.constant 32 : i32
! CHECK:               fir.result %[[VAL_20]] : i32
! CHECK:             }
! CHECK:             %[[VAL_21:.*]] = arith.constant 32 : i32
! CHECK:             %[[VAL_22:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_23:.*]] = arith.constant -1 : i32
! CHECK:             %[[VAL_24:.*]] = arith.constant 31 : i32
! CHECK:             %[[VAL_25:.*]] = arith.shrsi %[[VAL_13]], %[[VAL_24]] : i32
! CHECK:             %[[VAL_26:.*]] = arith.xori %[[VAL_13]], %[[VAL_25]] : i32
! CHECK:             %[[VAL_27:.*]] = arith.subi %[[VAL_26]], %[[VAL_25]] : i32
! CHECK:             %[[VAL_28:.*]] = arith.subi %[[VAL_29:.*]], %[[VAL_27]] : i32
! CHECK:             %[[VAL_30:.*]] = arith.cmpi eq, %[[VAL_13]], %[[VAL_22]] : i32
! CHECK:             %[[VAL_31:.*]] = arith.cmpi eq, %[[VAL_27]], %[[VAL_29]] : i32
! CHECK:             %[[VAL_32:.*]] = arith.ori %[[VAL_30]], %[[VAL_31]] : i1
! CHECK:             %[[VAL_33:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_22]] : i32
! CHECK:             %[[VAL_34:.*]] = arith.select %[[VAL_33]], %[[VAL_27]], %[[VAL_28]] : i32
! CHECK:             %[[VAL_35:.*]] = arith.select %[[VAL_33]], %[[VAL_28]], %[[VAL_27]] : i32
! CHECK:             %[[VAL_36:.*]] = arith.cmpi ne, %[[VAL_29]], %[[VAL_21]] : i32
! CHECK:             %[[VAL_37:.*]] = arith.shrui %[[VAL_17]], %[[VAL_29]] : i32
! CHECK:             %[[VAL_38:.*]] = arith.shli %[[VAL_37]], %[[VAL_29]] : i32
! CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_36]], %[[VAL_38]], %[[VAL_22]] : i32
! CHECK:             %[[VAL_40:.*]] = arith.subi %[[VAL_21]], %[[VAL_34]] : i32
! CHECK:             %[[VAL_41:.*]] = arith.shrui %[[VAL_23]], %[[VAL_40]] : i32
! CHECK:             %[[VAL_42:.*]] = arith.shrui %[[VAL_17]], %[[VAL_35]] : i32
! CHECK:             %[[VAL_43:.*]] = arith.andi %[[VAL_42]], %[[VAL_41]] : i32
! CHECK:             %[[VAL_44:.*]] = arith.subi %[[VAL_21]], %[[VAL_35]] : i32
! CHECK:             %[[VAL_45:.*]] = arith.shrui %[[VAL_23]], %[[VAL_44]] : i32
! CHECK:             %[[VAL_46:.*]] = arith.andi %[[VAL_17]], %[[VAL_45]] : i32
! CHECK:             %[[VAL_47:.*]] = arith.shli %[[VAL_46]], %[[VAL_34]] : i32
! CHECK:             %[[VAL_48:.*]] = arith.ori %[[VAL_39]], %[[VAL_43]] : i32
! CHECK:             %[[VAL_49:.*]] = arith.ori %[[VAL_48]], %[[VAL_47]] : i32
! CHECK:             %[[VAL_50:.*]] = arith.select %[[VAL_32]], %[[VAL_17]], %[[VAL_49]] : i32
! CHECK:             hlfir.yield_element %[[VAL_50]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_51:.*]] to %[[VAL_9]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
! CHECK:           hlfir.destroy %[[VAL_51]] : !hlfir.expr<42xi32>
! CHECK:           %[[VAL_52:.*]] = fir.load %[[VAL_9]]#1 : !fir.ref<!fir.array<42xi32>>
! CHECK:           return %[[VAL_52]] : !fir.array<42xi32>
! CHECK:         }

subroutine allocatables_test(a, b, c)
  implicit none
  integer, parameter :: nx = 1
  integer, parameter :: ny = 2
  integer, parameter :: nz = 3
  integer, dimension(:,:,:), allocatable :: a, b, c

  allocate(a(nx,ny,nz))
  allocate(b(nx,ny,nz))
  allocate(c(nx,ny,nz))

  c = min(a, b, c)
end subroutine
! CHECK-LABEL:   func.func @_QPallocatables_test(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>> {fir.bindc_name = "a"},
! CHECK-SAME:                                    %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>> {fir.bindc_name = "b"},
! CHECK-SAME:                                    %[[VAL_2:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFallocatables_testEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFallocatables_testEb"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFallocatables_testEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>)
! CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QFallocatables_testECnx) : !fir.ref<i32>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFallocatables_testECnx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QFallocatables_testECny) : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFallocatables_testECny"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QFallocatables_testECnz) : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFallocatables_testECnz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
! CHECK:           %[[VAL_14:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
! CHECK:           %[[VAL_16:.*]] = arith.constant 3 : i32
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> index
! CHECK:           %[[VAL_18:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_18]] : index
! CHECK:           %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_13]], %[[VAL_18]] : index
! CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_21]] : index
! CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_15]], %[[VAL_21]] : index
! CHECK:           %[[VAL_24:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_24]] : index
! CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_17]], %[[VAL_24]] : index
! CHECK:           %[[VAL_27:.*]] = fir.allocmem !fir.array<?x?x?xi32>, %[[VAL_20]], %[[VAL_23]], %[[VAL_26]] {fir.must_be_heap = true, uniq_name = "_QFallocatables_testEa.alloc"}
! CHECK:           %[[VAL_28:.*]] = fir.shape %[[VAL_20]], %[[VAL_23]], %[[VAL_26]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[VAL_29:.*]] = fir.embox %[[VAL_27]](%[[VAL_28]]) : (!fir.heap<!fir.array<?x?x?xi32>>, !fir.shape<3>) -> !fir.box<!fir.heap<!fir.array<?x?x?xi32>>>
! CHECK:           fir.store %[[VAL_29]] to %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
! CHECK:           %[[VAL_30:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> index
! CHECK:           %[[VAL_32:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> index
! CHECK:           %[[VAL_34:.*]] = arith.constant 3 : i32
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> index
! CHECK:           %[[VAL_36:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_37:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_36]] : index
! CHECK:           %[[VAL_38:.*]] = arith.select %[[VAL_37]], %[[VAL_31]], %[[VAL_36]] : index
! CHECK:           %[[VAL_39:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_40:.*]] = arith.cmpi sgt, %[[VAL_33]], %[[VAL_39]] : index
! CHECK:           %[[VAL_41:.*]] = arith.select %[[VAL_40]], %[[VAL_33]], %[[VAL_39]] : index
! CHECK:           %[[VAL_42:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_43:.*]] = arith.cmpi sgt, %[[VAL_35]], %[[VAL_42]] : index
! CHECK:           %[[VAL_44:.*]] = arith.select %[[VAL_43]], %[[VAL_35]], %[[VAL_42]] : index
! CHECK:           %[[VAL_45:.*]] = fir.allocmem !fir.array<?x?x?xi32>, %[[VAL_38]], %[[VAL_41]], %[[VAL_44]] {fir.must_be_heap = true, uniq_name = "_QFallocatables_testEb.alloc"}
! CHECK:           %[[VAL_46:.*]] = fir.shape %[[VAL_38]], %[[VAL_41]], %[[VAL_44]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[VAL_47:.*]] = fir.embox %[[VAL_45]](%[[VAL_46]]) : (!fir.heap<!fir.array<?x?x?xi32>>, !fir.shape<3>) -> !fir.box<!fir.heap<!fir.array<?x?x?xi32>>>
! CHECK:           fir.store %[[VAL_47]] to %[[VAL_4]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
! CHECK:           %[[VAL_48:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i32) -> index
! CHECK:           %[[VAL_50:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i32) -> index
! CHECK:           %[[VAL_52:.*]] = arith.constant 3 : i32
! CHECK:           %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i32) -> index
! CHECK:           %[[VAL_54:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_55:.*]] = arith.cmpi sgt, %[[VAL_49]], %[[VAL_54]] : index
! CHECK:           %[[VAL_56:.*]] = arith.select %[[VAL_55]], %[[VAL_49]], %[[VAL_54]] : index
! CHECK:           %[[VAL_57:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_58:.*]] = arith.cmpi sgt, %[[VAL_51]], %[[VAL_57]] : index
! CHECK:           %[[VAL_59:.*]] = arith.select %[[VAL_58]], %[[VAL_51]], %[[VAL_57]] : index
! CHECK:           %[[VAL_60:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_61:.*]] = arith.cmpi sgt, %[[VAL_53]], %[[VAL_60]] : index
! CHECK:           %[[VAL_62:.*]] = arith.select %[[VAL_61]], %[[VAL_53]], %[[VAL_60]] : index
! CHECK:           %[[VAL_63:.*]] = fir.allocmem !fir.array<?x?x?xi32>, %[[VAL_56]], %[[VAL_59]], %[[VAL_62]] {fir.must_be_heap = true, uniq_name = "_QFallocatables_testEc.alloc"}
! CHECK:           %[[VAL_64:.*]] = fir.shape %[[VAL_56]], %[[VAL_59]], %[[VAL_62]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[VAL_65:.*]] = fir.embox %[[VAL_63]](%[[VAL_64]]) : (!fir.heap<!fir.array<?x?x?xi32>>, !fir.shape<3>) -> !fir.box<!fir.heap<!fir.array<?x?x?xi32>>>
! CHECK:           fir.store %[[VAL_65]] to %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
! CHECK:           %[[VAL_66:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
! CHECK:           %[[VAL_67:.*]] = fir.box_addr %[[VAL_66]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>) -> !fir.heap<!fir.array<?x?x?xi32>>
! CHECK:           %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (!fir.heap<!fir.array<?x?x?xi32>>) -> i64
! CHECK:           %[[VAL_69:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_70:.*]] = arith.cmpi ne, %[[VAL_68]], %[[VAL_69]] : i64
! CHECK:           %[[VAL_71:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
! CHECK:           %[[VAL_72:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_73:.*]]:3 = fir.box_dims %[[VAL_71]], %[[VAL_72]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_74:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_75:.*]]:3 = fir.box_dims %[[VAL_71]], %[[VAL_74]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_76:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_77:.*]]:3 = fir.box_dims %[[VAL_71]], %[[VAL_76]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_78:.*]] = fir.shape %[[VAL_73]]#1, %[[VAL_75]]#1, %[[VAL_77]]#1 : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[VAL_79:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
! CHECK:           %[[VAL_80:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
! CHECK:           %[[VAL_81:.*]] = hlfir.elemental %[[VAL_78]] unordered : (!fir.shape<3>) -> !hlfir.expr<?x?x?xi32> {
! CHECK:           ^bb0(%[[VAL_82:.*]]: index, %[[VAL_83:.*]]: index, %[[VAL_84:.*]]: index):
! CHECK:             %[[VAL_85:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_86:.*]]:3 = fir.box_dims %[[VAL_71]], %[[VAL_85]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_87:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_88:.*]]:3 = fir.box_dims %[[VAL_71]], %[[VAL_87]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_89:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_90:.*]]:3 = fir.box_dims %[[VAL_71]], %[[VAL_89]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_91:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_92:.*]] = arith.subi %[[VAL_86]]#0, %[[VAL_91]] : index
! CHECK:             %[[VAL_93:.*]] = arith.addi %[[VAL_82]], %[[VAL_92]] : index
! CHECK:             %[[VAL_94:.*]] = arith.subi %[[VAL_88]]#0, %[[VAL_91]] : index
! CHECK:             %[[VAL_95:.*]] = arith.addi %[[VAL_83]], %[[VAL_94]] : index
! CHECK:             %[[VAL_96:.*]] = arith.subi %[[VAL_90]]#0, %[[VAL_91]] : index
! CHECK:             %[[VAL_97:.*]] = arith.addi %[[VAL_84]], %[[VAL_96]] : index
! CHECK:             %[[VAL_98:.*]] = hlfir.designate %[[VAL_71]] (%[[VAL_93]], %[[VAL_95]], %[[VAL_97]])  : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index, index, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_99:.*]] = fir.load %[[VAL_98]] : !fir.ref<i32>
! CHECK:             %[[VAL_100:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_101:.*]]:3 = fir.box_dims %[[VAL_79]], %[[VAL_100]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_102:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_103:.*]]:3 = fir.box_dims %[[VAL_79]], %[[VAL_102]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_104:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_105:.*]]:3 = fir.box_dims %[[VAL_79]], %[[VAL_104]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_106:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_107:.*]] = arith.subi %[[VAL_101]]#0, %[[VAL_106]] : index
! CHECK:             %[[VAL_108:.*]] = arith.addi %[[VAL_82]], %[[VAL_107]] : index
! CHECK:             %[[VAL_109:.*]] = arith.subi %[[VAL_103]]#0, %[[VAL_106]] : index
! CHECK:             %[[VAL_110:.*]] = arith.addi %[[VAL_83]], %[[VAL_109]] : index
! CHECK:             %[[VAL_111:.*]] = arith.subi %[[VAL_105]]#0, %[[VAL_106]] : index
! CHECK:             %[[VAL_112:.*]] = arith.addi %[[VAL_84]], %[[VAL_111]] : index
! CHECK:             %[[VAL_113:.*]] = hlfir.designate %[[VAL_79]] (%[[VAL_108]], %[[VAL_110]], %[[VAL_112]])  : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index, index, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_114:.*]] = fir.load %[[VAL_113]] : !fir.ref<i32>
! CHECK:             %[[VAL_115:.*]] = arith.cmpi slt, %[[VAL_99]], %[[VAL_114]] : i32
! CHECK:             %[[VAL_116:.*]] = arith.select %[[VAL_115]], %[[VAL_99]], %[[VAL_114]] : i32
! CHECK:             %[[VAL_117:.*]] = fir.if %[[VAL_70]] -> (i32) {
! CHECK:               %[[VAL_118:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_119:.*]]:3 = fir.box_dims %[[VAL_80]], %[[VAL_118]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:               %[[VAL_120:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_121:.*]]:3 = fir.box_dims %[[VAL_80]], %[[VAL_120]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:               %[[VAL_122:.*]] = arith.constant 2 : index
! CHECK:               %[[VAL_123:.*]]:3 = fir.box_dims %[[VAL_80]], %[[VAL_122]] : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index) -> (index, index, index)
! CHECK:               %[[VAL_124:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_125:.*]] = arith.subi %[[VAL_119]]#0, %[[VAL_124]] : index
! CHECK:               %[[VAL_126:.*]] = arith.addi %[[VAL_82]], %[[VAL_125]] : index
! CHECK:               %[[VAL_127:.*]] = arith.subi %[[VAL_121]]#0, %[[VAL_124]] : index
! CHECK:               %[[VAL_128:.*]] = arith.addi %[[VAL_83]], %[[VAL_127]] : index
! CHECK:               %[[VAL_129:.*]] = arith.subi %[[VAL_123]]#0, %[[VAL_124]] : index
! CHECK:               %[[VAL_130:.*]] = arith.addi %[[VAL_84]], %[[VAL_129]] : index
! CHECK:               %[[VAL_131:.*]] = hlfir.designate %[[VAL_80]] (%[[VAL_126]], %[[VAL_128]], %[[VAL_130]])  : (!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>, index, index, index) -> !fir.ref<i32>
! CHECK:               %[[VAL_132:.*]] = fir.load %[[VAL_131]] : !fir.ref<i32>
! CHECK:               %[[VAL_133:.*]] = arith.cmpi slt, %[[VAL_116]], %[[VAL_132]] : i32
! CHECK:               %[[VAL_134:.*]] = arith.select %[[VAL_133]], %[[VAL_116]], %[[VAL_132]] : i32
! CHECK:               fir.result %[[VAL_134]] : i32
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_116]] : i32
! CHECK:             }
! CHECK:             hlfir.yield_element %[[VAL_135:.*]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_136:.*]] to %[[VAL_5]]#0 realloc : !hlfir.expr<?x?x?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xi32>>>>
! CHECK:           hlfir.destroy %[[VAL_136]] : !hlfir.expr<?x?x?xi32>
! CHECK:           return
! CHECK:         }