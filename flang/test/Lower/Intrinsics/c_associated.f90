! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! Test intrinsic module procedure c_associated

! CHECK-LABEL: func.func @_QPtest_c_ptr(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr1"},
! CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr2"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.logical<4> {bindc_name = "z1", uniq_name = "_QFtest_c_ptrEz1"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.logical<4> {bindc_name = "z2", uniq_name = "_QFtest_c_ptrEz2"}
! CHECK:         %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:         %[[VAL_10:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_10]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<i64>
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> i64
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_16]] : i64
! CHECK:         %[[VAL_18:.*]] = fir.if %[[VAL_17]] -> (i1) {
! CHECK:           fir.result %[[VAL_14]] : i1
! CHECK:         } else {
! CHECK:           %[[VAL_19:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_19]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i64>
! CHECK:           %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_21]] : i64
! CHECK:           %[[VAL_23:.*]] = arith.andi %[[VAL_14]], %[[VAL_22]] : i1
! CHECK:           fir.result %[[VAL_23]] : i1
! CHECK:         }
! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_25:.*]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_24]] to %[[VAL_3]] : !fir.ref<!fir.logical<4>>
! CHECK:         return
! CHECK:       }

subroutine test_c_ptr(cptr1, cptr2)
  use iso_c_binding
  type(c_ptr) :: cptr1, cptr2
  logical :: z1, z2

  z1 = c_associated(cptr1)

  z2 = c_associated(cptr1, cptr2)
end

! CHECK-LABEL: func.func @_QPtest_c_funptr(
! CHECK-SAME:                              %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "cptr1"},
! CHECK-SAME:                              %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "cptr2"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.logical<4> {bindc_name = "z1", uniq_name = "_QFtest_c_funptrEz1"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.logical<4> {bindc_name = "z2", uniq_name = "_QFtest_c_funptrEz2"}
! CHECK:         %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:         %[[VAL_10:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_10]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<i64>
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>) -> i64
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_16]] : i64
! CHECK:         %[[VAL_18:.*]] = fir.if %[[VAL_17]] -> (i1) {
! CHECK:           fir.result %[[VAL_14]] : i1
! CHECK:         } else {
! CHECK:           %[[VAL_19:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_19]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i64>
! CHECK:           %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_21]] : i64
! CHECK:           %[[VAL_23:.*]] = arith.andi %[[VAL_14]], %[[VAL_22]] : i1
! CHECK:           fir.result %[[VAL_23]] : i1
! CHECK:         }
! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_25:.*]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_24]] to %[[VAL_3]] : !fir.ref<!fir.logical<4>>
! CHECK:         return
! CHECK:       }

subroutine test_c_funptr(cptr1, cptr2)
  use iso_c_binding
  type(c_funptr) :: cptr1, cptr2
  logical :: z1, z2

  z1 = c_associated(cptr1)

  z2 = c_associated(cptr1, cptr2)
end

! CHECK-LABEL: func.func @_QPtest_optional_argument(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr1"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr2", fir.optional},
! CHECK-SAME:                                       %[[VAL_2:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "cfunptr1"},
! CHECK-SAME:                                       %[[VAL_3:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "cfunptr2", fir.optional}) {
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.logical<4> {bindc_name = "z1", uniq_name = "_QFtest_optional_argumentEz1"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.logical<4> {bindc_name = "z2", uniq_name = "_QFtest_optional_argumentEz2"}
! CHECK:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i64>
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_9]] : i64
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> i64
! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:         %[[VAL_14:.*]] = fir.if %[[VAL_13]] -> (i1) {
! CHECK:           fir.result %[[VAL_10]] : i1
! CHECK:         } else {
! CHECK:           %[[VAL_15:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_15]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i64>
! CHECK:           %[[VAL_18:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_17]] : i64
! CHECK:           %[[VAL_19:.*]] = arith.andi %[[VAL_10]], %[[VAL_18]] : i1
! CHECK:           fir.result %[[VAL_19]] : i1
! CHECK:         }
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_21:.*]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_20]] to %[[VAL_4]] : !fir.ref<!fir.logical<4>>
! CHECK:         %[[VAL_22:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_23:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_22]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i64>
! CHECK:         %[[VAL_25:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_26:.*]] = arith.cmpi ne, %[[VAL_24]], %[[VAL_25]] : i64
! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>) -> i64
! CHECK:         %[[VAL_28:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_29:.*]] = arith.cmpi eq, %[[VAL_27]], %[[VAL_28]] : i64
! CHECK:         %[[VAL_30:.*]] = fir.if %[[VAL_29]] -> (i1) {
! CHECK:           fir.result %[[VAL_26]] : i1
! CHECK:         } else {
! CHECK:           %[[VAL_31:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:           %[[VAL_32:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_31]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_32]] : !fir.ref<i64>
! CHECK:           %[[VAL_34:.*]] = arith.cmpi eq, %[[VAL_24]], %[[VAL_33]] : i64
! CHECK:           %[[VAL_35:.*]] = arith.andi %[[VAL_26]], %[[VAL_34]] : i1
! CHECK:           fir.result %[[VAL_35]] : i1
! CHECK:         }
! CHECK:         %[[VAL_36:.*]] = fir.convert %[[VAL_37:.*]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_36]] to %[[VAL_5]] : !fir.ref<!fir.logical<4>>
! CHECK:         return
! CHECK:       }

subroutine test_optional_argument(cptr1, cptr2, cfunptr1, cfunptr2)
  use iso_c_binding
  type(c_ptr) :: cptr1
  type(c_ptr), optional :: cptr2
  type(c_funptr) :: cfunptr1
  type(c_funptr), optional :: cfunptr2
  logical :: z1, z2

  z1 = c_associated(cptr1, cptr2)

  z2 = c_associated(cfunptr1, cfunptr2)
end
