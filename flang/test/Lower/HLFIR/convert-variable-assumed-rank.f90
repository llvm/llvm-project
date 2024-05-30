! Test lowering of assumed-rank variables
! RUN: bbc -emit-hlfir %s -allow-assumed-rank -o - | FileCheck %s

module assumed_rank_tests
interface
subroutine takes_real(x)
  real :: x(..)
end subroutine
subroutine takes_char(x)
  character(*) :: x(..)
end subroutine
end interface
contains

subroutine test_intrinsic(x)
  real :: x(..)
  call takes_real(x)
end subroutine

subroutine test_character_explicit_len(x, n)
  integer(8) :: n
  character(n) :: x(..)
  call takes_char(x)
end subroutine

subroutine test_character_assumed_len(x)
  character(*) :: x(..)
  call takes_char(x)
end subroutine

subroutine test_with_attrs(x)
  real, target, optional :: x(..)
  call takes_real(x)
end subroutine
! CHECK-LABEL:   func.func @_QMassumed_rank_testsPtest_intrinsic(
! CHECK-SAME:                                                    %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QMassumed_rank_testsFtest_intrinsicEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPtakes_real(%[[VAL_2]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMassumed_rank_testsPtest_character_explicit_len(
! CHECK-SAME:                                                                 %[[VAL_0:.*]]: !fir.box<!fir.array<*:!fir.char<1,?>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                                 %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QMassumed_rank_testsFtest_character_explicit_lenEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_7]] dummy_scope %[[VAL_2]] {uniq_name = "_QMassumed_rank_testsFtest_character_explicit_lenEx"} : (!fir.box<!fir.array<*:!fir.char<1,?>>>, i64, !fir.dscope) -> (!fir.box<!fir.array<*:!fir.char<1,?>>>, !fir.box<!fir.array<*:!fir.char<1,?>>>)
! CHECK:           fir.call @_QPtakes_char(%[[VAL_8]]#0) fastmath<contract> : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMassumed_rank_testsPtest_character_assumed_len(
! CHECK-SAME:                                                                %[[VAL_0:.*]]: !fir.box<!fir.array<*:!fir.char<1,?>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QMassumed_rank_testsFtest_character_assumed_lenEx"} : (!fir.box<!fir.array<*:!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<*:!fir.char<1,?>>>, !fir.box<!fir.array<*:!fir.char<1,?>>>)
! CHECK:           fir.call @_QPtakes_char(%[[VAL_2]]#0) fastmath<contract> : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMassumed_rank_testsPtest_with_attrs(
! CHECK-SAME:                                                     %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x", fir.optional, fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<optional, target>, uniq_name = "_QMassumed_rank_testsFtest_with_attrsEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPtakes_real(%[[VAL_2]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
end module
