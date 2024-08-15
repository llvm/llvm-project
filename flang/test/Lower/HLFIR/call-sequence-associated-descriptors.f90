! Test lowering of sequence associated arguments (F'2023 15.5.2.12) passed
! by descriptor. The descriptor on the caller side is prepared according to
! the dummy argument shape.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module bindc_seq_assoc
  interface
    subroutine takes_char(x, n) bind(c)
      integer :: n
      character(*) :: x(n)
    end subroutine
    subroutine takes_char_assumed_size(x) bind(c)
      character(*) :: x(10, *)
    end subroutine
    subroutine takes_optional_char(x, n) bind(c)
      integer :: n
      character(*), optional :: x(n)
    end subroutine
  end interface
contains
  subroutine test_char_1(x)
    character(*) :: x(10, 20)
    call takes_char(x, 100)
  end subroutine
! CHECK-LABEL:   func.func @_QMbindc_seq_assocPtest_char_1(
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_2:.*]](%[[VAL_5:.*]]) typeparams %[[VAL_1:.*]]#1 dummy_scope %{{[0-9]+}} {uniq_name = "_QMbindc_seq_assocFtest_char_1Ex"} : (!fir.ref<!fir.array<10x20x!fir.char<1,?>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<10x20x!fir.char<1,?>>>, !fir.ref<!fir.array<10x20x!fir.char<1,?>>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 100 : i32
! CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_9:.*]] = fir.shift %[[VAL_8]], %[[VAL_8]] : (index, index) -> !fir.shift<2>
! CHECK:           %[[VAL_10:.*]] = fir.rebox %[[VAL_6]]#0(%[[VAL_9]]) : (!fir.box<!fir.array<10x20x!fir.char<1,?>>>, !fir.shift<2>) -> !fir.box<!fir.array<10x20x!fir.char<1,?>>>
! CHECK:           %[[VAL_11:.*]]:3 = hlfir.associate %[[VAL_7]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]]#1 {uniq_name = "_QMbindc_seq_assocFtakes_charEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> i64
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_17:.*]] = arith.subi %[[VAL_15]], %[[VAL_16]] : i64
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_18]] : i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_20]], %[[VAL_21]] : index
! CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_20]], %[[VAL_21]] : index
! CHECK:           %[[VAL_24:.*]] = fir.shape_shift %[[VAL_12]], %[[VAL_23]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_25:.*]] = fir.box_elesize %[[VAL_10]] : (!fir.box<!fir.array<10x20x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_26:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.array<10x20x!fir.char<1,?>>>) -> !fir.ref<!fir.array<10x20x!fir.char<1,?>>>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.array<10x20x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:           %[[VAL_28:.*]] = fir.embox %[[VAL_27]](%[[VAL_24]]) typeparams %[[VAL_25]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shapeshift<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           fir.call @takes_char(%[[VAL_28]], %[[VAL_11]]#1) fastmath<contract> {is_bind_c} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_11]]#1, %[[VAL_11]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

  subroutine test_char_copy_in_copy_out(x)
    character(*) :: x(:, :)
    call takes_char(x, 100)
  end subroutine
! CHECK-LABEL:   func.func @_QMbindc_seq_assocPtest_char_copy_in_copy_out(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:.*]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMbindc_seq_assocFtest_char_copy_in_copy_outEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 100 : i32
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>>) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i1)
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]] = fir.shift %[[VAL_4]], %[[VAL_4]] : (index, index) -> !fir.shift<2>
! CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_3]]#0(%[[VAL_5]]) : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:           %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_2]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]]#1 {uniq_name = "_QMbindc_seq_assocFtakes_charEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
! CHECK:           %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_17]] : index
! CHECK:           %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_16]], %[[VAL_17]] : index
! CHECK:           %[[VAL_20:.*]] = fir.shape_shift %[[VAL_8]], %[[VAL_19]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_21:.*]] = fir.box_elesize %[[VAL_6]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_22:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:           %[[VAL_24:.*]] = fir.embox %[[VAL_23]](%[[VAL_20]]) typeparams %[[VAL_21]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shapeshift<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           fir.call @takes_char(%[[VAL_24]], %[[VAL_7]]#1) fastmath<contract> {is_bind_c} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_3]]#1 to %[[VAL_1]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>>, i1, !fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

  subroutine test_char_assumed_size(x)
    character(*) :: x(:, :)
    call takes_char_assumed_size(x)
  end subroutine
! CHECK-LABEL:   func.func @_QMbindc_seq_assocPtest_char_assumed_size(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:.*]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMbindc_seq_assocFtest_char_assumed_sizeEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>>) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i1)
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]] = fir.shift %[[VAL_3]], %[[VAL_3]] : (index, index) -> !fir.shift<2>
! CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_2]]#0(%[[VAL_4]]) : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : i64
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_13]] : index
! CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_13]] : index
! CHECK:           %[[VAL_16:.*]] = arith.constant -1 : i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = fir.shape_shift %[[VAL_6]], %[[VAL_15]], %[[VAL_6]], %[[VAL_17]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_19:.*]] = fir.box_elesize %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_20:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<10x?x!fir.char<1,?>>>
! CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_21]](%[[VAL_18]]) typeparams %[[VAL_19]] : (!fir.ref<!fir.array<10x?x!fir.char<1,?>>>, !fir.shapeshift<2>, index) -> !fir.box<!fir.array<10x?x!fir.char<1,?>>>
! CHECK:           fir.call @takes_char_assumed_size(%[[VAL_22]]) fastmath<contract> {is_bind_c} : (!fir.box<!fir.array<10x?x!fir.char<1,?>>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_2]]#1 to %[[VAL_1]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>>, i1, !fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> ()
! CHECK:           return
! CHECK:         }

  subroutine test_optional_char(x)
    character(*), optional :: x(10, 20)
    call takes_optional_char(x, 100)
  end subroutine
! CHECK-LABEL:   func.func @_QMbindc_seq_assocPtest_optional_char(
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_2:.*]](%[[VAL_5:.*]]) typeparams %[[VAL_1:.*]]#1 dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMbindc_seq_assocFtest_optional_charEx"} : (!fir.ref<!fir.array<10x20x!fir.char<1,?>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<10x20x!fir.char<1,?>>>, !fir.ref<!fir.array<10x20x!fir.char<1,?>>>)
! CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_6]]#0 : (!fir.box<!fir.array<10x20x!fir.char<1,?>>>) -> i1
! CHECK:           %[[VAL_8:.*]] = arith.constant 100 : i32
! CHECK:           %[[VAL_9:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<10x20x!fir.char<1,?>>>) {
! CHECK:             %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_11:.*]] = fir.shift %[[VAL_10]], %[[VAL_10]] : (index, index) -> !fir.shift<2>
! CHECK:             %[[VAL_12:.*]] = fir.rebox %[[VAL_6]]#0(%[[VAL_11]]) : (!fir.box<!fir.array<10x20x!fir.char<1,?>>>, !fir.shift<2>) -> !fir.box<!fir.array<10x20x!fir.char<1,?>>>
! CHECK:             fir.result %[[VAL_12]] : !fir.box<!fir.array<10x20x!fir.char<1,?>>>
! CHECK:           } else {
! CHECK:             %[[VAL_13:.*]] = fir.absent !fir.box<!fir.array<10x20x!fir.char<1,?>>>
! CHECK:             fir.result %[[VAL_13]] : !fir.box<!fir.array<10x20x!fir.char<1,?>>>
! CHECK:           }
! CHECK:           %[[VAL_14:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           %[[VAL_15:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:             %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_14]]#1 {uniq_name = "_QMbindc_seq_assocFtakes_optional_charEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_18:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_21:.*]] = arith.subi %[[VAL_19]], %[[VAL_20]] : i64
! CHECK:             %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : i64
! CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
! CHECK:             %[[VAL_25:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_26:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_25]] : index
! CHECK:             %[[VAL_27:.*]] = arith.select %[[VAL_26]], %[[VAL_24]], %[[VAL_25]] : index
! CHECK:             %[[VAL_28:.*]] = fir.shape_shift %[[VAL_16]], %[[VAL_27]] : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_29:.*]] = fir.box_elesize %[[VAL_9]] : (!fir.box<!fir.array<10x20x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.array<10x20x!fir.char<1,?>>>) -> !fir.ref<!fir.array<10x20x!fir.char<1,?>>>
! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.array<10x20x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:             %[[VAL_32:.*]] = fir.embox %[[VAL_31]](%[[VAL_28]]) typeparams %[[VAL_29]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shapeshift<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:             fir.result %[[VAL_32]] : !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           } else {
! CHECK:             %[[VAL_33:.*]] = fir.absent !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:             fir.result %[[VAL_33]] : !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           }
! CHECK:           fir.call @takes_optional_char(%[[VAL_15]], %[[VAL_14]]#1) fastmath<contract> {is_bind_c} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_14]]#1, %[[VAL_14]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }
end module

module poly_seq_assoc
  interface
    subroutine takes_poly(x, n)
      integer :: n
      class(*) :: x(n)
    end subroutine
    subroutine takes_poly_assumed_size(x)
      class(*) :: x(10, *)
    end subroutine
    subroutine takes_optional_poly(x, n)
      integer :: n
      class(*), optional :: x(n)
    end subroutine
  end interface
contains
  subroutine test_poly_1(x)
    class(*) :: x(10, 20)
    call takes_poly(x, 100)
  end subroutine
! CHECK-LABEL:   func.func @_QMpoly_seq_assocPtest_poly_1(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.class<!fir.array<10x20xnone>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:.*]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMpoly_seq_assocFtest_poly_1Ex"} : (!fir.class<!fir.array<10x20xnone>>, !fir.dscope) -> (!fir.class<!fir.array<10x20xnone>>, !fir.class<!fir.array<10x20xnone>>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 100 : i32
! CHECK:           %[[VAL_3:.*]]:3 = hlfir.associate %[[VAL_2]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]]#1 {uniq_name = "_QMpoly_seq_assocFtakes_polyEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : index
! CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_11]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_16:.*]] = fir.box_addr %[[VAL_1]]#1 : (!fir.class<!fir.array<10x20xnone>>) -> !fir.ref<!fir.array<10x20xnone>>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.array<10x20xnone>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_17]](%[[VAL_15]]) source_box %[[VAL_1]]#0 : (!fir.ref<!fir.array<?xnone>>, !fir.shape<1>, !fir.class<!fir.array<10x20xnone>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:           fir.call @_QPtakes_poly(%[[VAL_18]], %[[VAL_3]]#1) fastmath<contract> : (!fir.class<!fir.array<?xnone>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_3]]#1, %[[VAL_3]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

  subroutine test_poly_copy_in_copy_out(x)
    class(*) :: x(:, :)
    call takes_poly(x, 100)
  end subroutine
! CHECK-LABEL:   func.func @_QMpoly_seq_assocPtest_poly_copy_in_copy_out(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:.*]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMpoly_seq_assocFtest_poly_copy_in_copy_outEx"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> (!fir.class<!fir.array<?x?xnone>>, !fir.class<!fir.array<?x?xnone>>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 100 : i32
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 to %[[TMP_BOX:.*]] : (!fir.class<!fir.array<?x?xnone>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> (!fir.class<!fir.array<?x?xnone>>, i1)
! CHECK:           %[[VAL_4:.*]]:3 = hlfir.associate %[[VAL_2]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#1 {uniq_name = "_QMpoly_seq_assocFtakes_polyEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> i64
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_13]] : index
! CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_13]] : index
! CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_17:.*]] = fir.box_addr %[[VAL_3]]#0 : (!fir.class<!fir.array<?x?xnone>>) -> !fir.ref<!fir.array<?x?xnone>>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.array<?x?xnone>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_16]]) source_box %[[VAL_3]]#0 : (!fir.ref<!fir.array<?xnone>>, !fir.shape<1>, !fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:           fir.call @_QPtakes_poly(%[[VAL_19]], %[[VAL_4]]#1) fastmath<contract> : (!fir.class<!fir.array<?xnone>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_3]]#1 to %[[VAL_1]]#0 : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, i1, !fir.class<!fir.array<?x?xnone>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_4]]#1, %[[VAL_4]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

  subroutine test_poly_assumed_size(x)
    class(*) :: x(:, :)
    call takes_poly_assumed_size(x)
  end subroutine
! CHECK-LABEL:   func.func @_QMpoly_seq_assocPtest_poly_assumed_size(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:.*]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMpoly_seq_assocFtest_poly_assumed_sizeEx"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> (!fir.class<!fir.array<?x?xnone>>, !fir.class<!fir.array<?x?xnone>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 to %[[TMP_BOX:.*]] : (!fir.class<!fir.array<?x?xnone>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> (!fir.class<!fir.array<?x?xnone>>, i1)
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : index
! CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_9]] : index
! CHECK:           %[[VAL_12:.*]] = arith.constant -1 : i64
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_11]], %[[VAL_13]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_15:.*]] = fir.box_addr %[[VAL_2]]#0 : (!fir.class<!fir.array<?x?xnone>>) -> !fir.ref<!fir.array<?x?xnone>>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.array<?x?xnone>>) -> !fir.ref<!fir.array<10x?xnone>>
! CHECK:           %[[VAL_17:.*]] = fir.embox %[[VAL_16]](%[[VAL_14]]) source_box %[[VAL_2]]#0 : (!fir.ref<!fir.array<10x?xnone>>, !fir.shape<2>, !fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.array<10x?xnone>>
! CHECK:           fir.call @_QPtakes_poly_assumed_size(%[[VAL_17]]) fastmath<contract> : (!fir.class<!fir.array<10x?xnone>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_2]]#1 to %[[VAL_1]]#0 : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, i1, !fir.class<!fir.array<?x?xnone>>) -> ()
! CHECK:           return
! CHECK:         }

  subroutine test_optional_poly(x)
    class(*), optional :: x(10, 20)
    call takes_optional_poly(x, 100)
  end subroutine
! CHECK-LABEL:   func.func @_QMpoly_seq_assocPtest_optional_poly(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:.*]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMpoly_seq_assocFtest_optional_polyEx"} : (!fir.class<!fir.array<10x20xnone>>, !fir.dscope) -> (!fir.class<!fir.array<10x20xnone>>, !fir.class<!fir.array<10x20xnone>>)
! CHECK:           %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.class<!fir.array<10x20xnone>>) -> i1
! CHECK:           %[[VAL_3:.*]] = arith.constant 100 : i32
! CHECK:           %[[VAL_4:.*]] = fir.if %[[VAL_2]] -> (!fir.class<!fir.array<10x20xnone>>) {
! CHECK:             fir.result %[[VAL_1]]#0 : !fir.class<!fir.array<10x20xnone>>
! CHECK:           } else {
! CHECK:             %[[VAL_5:.*]] = fir.absent !fir.class<!fir.array<10x20xnone>>
! CHECK:             fir.result %[[VAL_5]] : !fir.class<!fir.array<10x20xnone>>
! CHECK:           }
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_3]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           %[[VAL_7:.*]] = fir.if %[[VAL_2]] -> (!fir.class<!fir.array<?xnone>>) {
! CHECK:             %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]]#1 {uniq_name = "_QMpoly_seq_assocFtakes_optional_polyEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
! CHECK:             %[[VAL_11:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:             %[[VAL_13:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:             %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_16]] : index
! CHECK:             %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_15]], %[[VAL_16]] : index
! CHECK:             %[[VAL_19:.*]] = fir.shape %[[VAL_18]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_20:.*]] = fir.box_addr %[[VAL_4]] : (!fir.class<!fir.array<10x20xnone>>) -> !fir.ref<!fir.array<10x20xnone>>
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.array<10x20xnone>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK:             %[[VAL_22:.*]] = fir.embox %[[VAL_21]](%[[VAL_19]]) source_box %[[VAL_4]] : (!fir.ref<!fir.array<?xnone>>, !fir.shape<1>, !fir.class<!fir.array<10x20xnone>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:             fir.result %[[VAL_22]] : !fir.class<!fir.array<?xnone>>
! CHECK:           } else {
! CHECK:             %[[VAL_23:.*]] = fir.absent !fir.class<!fir.array<?xnone>>
! CHECK:             fir.result %[[VAL_23]] : !fir.class<!fir.array<?xnone>>
! CHECK:           }
! CHECK:           fir.call @_QPtakes_optional_poly(%[[VAL_7]], %[[VAL_6]]#1) fastmath<contract> : (!fir.class<!fir.array<?xnone>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }
end module
