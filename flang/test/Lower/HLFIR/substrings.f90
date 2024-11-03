! Test lowering of substrings to HLFIR
! Note: cse is run to make the expected output more readable by sharing
! the boilerplate between the different susbtring cases.
! RUN: bbc -emit-hlfir -o - %s | fir-opt -cse -o - | FileCheck %s

! CHECK-LABEL:   func.func @_QPcst_len(
subroutine cst_len(array, scalar)
  character(10) :: array(100), scalar
! CHECK:  %[[VAL_5:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_6:.*]]) typeparams %[[VAL_3:[^ ]*]] {{.*}}array"
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_3]] {{.*}}scalar"
  print *, array(:)(2:5)
! CHECK:  %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_16:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_17:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_18:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_19:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_15]]:%[[VAL_5]]:%[[VAL_15]]) substr %[[VAL_16]], %[[VAL_17]]  shape %[[VAL_6]] typeparams %[[VAL_18]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<100x!fir.char<1,4>>>

  print *, array(42)(2:5)
! CHECK:  %[[VAL_25:.*]] = arith.constant 42 : index
! CHECK:  %[[VAL_26:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_25]]) substr %[[VAL_16]], %[[VAL_17]]  typeparams %[[VAL_18]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index, index, index) -> !fir.ref<!fir.char<1,4>>
  print *, array(:)(2:)
! CHECK:  %[[VAL_33:.*]] = arith.constant 9 : index
! CHECK:  %[[VAL_34:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_15]]:%[[VAL_5]]:%[[VAL_15]]) substr %[[VAL_16]], %[[VAL_3]]  shape %[[VAL_6]] typeparams %[[VAL_33]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<100x!fir.char<1,9>>>

  print *, scalar(2:5)
! CHECK:  %[[VAL_40:.*]] = hlfir.designate %[[VAL_9]]#0  substr %[[VAL_16]], %[[VAL_17]]  typeparams %[[VAL_18]] : (!fir.ref<!fir.char<1,10>>, index, index, index) -> !fir.ref<!fir.char<1,4>>
end subroutine

! CHECK-LABEL:   func.func @_QPdyn_len(
subroutine dyn_len(array, scalar, l, n, m, k)
  integer(8) :: n,m,k
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare {{.*}}k"
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}m"
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}n"
  character(l) :: array(:), scalar
! CHECK:  %[[VAL_14:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_13:[^ ]*]] {{.*}}array"
! CHECK:  %[[VAL_19:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_18:[^ ]*]] {{.*}}scalar"

  print *, array(:)(n:m)
! CHECK:  %[[VAL_25:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_26:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_14]]#1, %[[VAL_26]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_28:.*]] = arith.subi %[[VAL_27]]#1, %[[VAL_25]] : index
! CHECK:  %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_25]] : index
! CHECK:  %[[VAL_30:.*]] = arith.divsi %[[VAL_29]], %[[VAL_25]] : index
! CHECK:  %[[VAL_31:.*]] = arith.cmpi sgt, %[[VAL_30]], %[[VAL_26]] : index
! CHECK:  %[[VAL_32:.*]] = arith.select %[[VAL_31]], %[[VAL_30]], %[[VAL_26]] : index
! CHECK:  %[[VAL_33:.*]] = fir.shape %[[VAL_32]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_34:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_35:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
! CHECK:  %[[VAL_37:.*]] = fir.convert %[[VAL_35]] : (i64) -> index
! CHECK:  %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_36]] : index
! CHECK:  %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_25]] : index
! CHECK:  %[[VAL_40:.*]] = arith.cmpi sgt, %[[VAL_39]], %[[VAL_26]] : index
! CHECK:  %[[VAL_41:.*]] = arith.select %[[VAL_40]], %[[VAL_39]], %[[VAL_26]] : index
! CHECK:  %[[VAL_42:.*]] = hlfir.designate %[[VAL_14]]#0 (%[[VAL_25]]:%[[VAL_27]]#1:%[[VAL_25]]) substr %[[VAL_36]], %[[VAL_37]]  shape %[[VAL_33]] typeparams %[[VAL_41]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>

  print *, array(k)(n:m)
! CHECK:  %[[VAL_48:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_49:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_50:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_51:.*]] = fir.convert %[[VAL_49]] : (i64) -> index
! CHECK:  %[[VAL_52:.*]] = fir.convert %[[VAL_50]] : (i64) -> index
! CHECK:  %[[VAL_53:.*]] = arith.subi %[[VAL_52]], %[[VAL_51]] : index
! CHECK:  %[[VAL_54:.*]] = arith.addi %[[VAL_53]], %[[VAL_25]] : index
! CHECK:  %[[VAL_55:.*]] = arith.cmpi sgt, %[[VAL_54]], %[[VAL_26]] : index
! CHECK:  %[[VAL_56:.*]] = arith.select %[[VAL_55]], %[[VAL_54]], %[[VAL_26]] : index
! CHECK:  %[[VAL_57:.*]] = hlfir.designate %[[VAL_14]]#0 (%[[VAL_48]]) substr %[[VAL_51]], %[[VAL_52]]  typeparams %[[VAL_56]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i64, index, index, index) -> !fir.boxchar<1>

  print *, array(:)(n:)
! CHECK:  %[[VAL_65:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_66:.*]] = fir.convert %[[VAL_13]] : (i32) -> i64
! CHECK:  %[[VAL_67:.*]] = fir.convert %[[VAL_65]] : (i64) -> index
! CHECK:  %[[VAL_68:.*]] = fir.convert %[[VAL_66]] : (i64) -> index
! CHECK:  %[[VAL_69:.*]] = arith.subi %[[VAL_68]], %[[VAL_67]] : index
! CHECK:  %[[VAL_70:.*]] = arith.addi %[[VAL_69]], %[[VAL_25]] : index
! CHECK:  %[[VAL_71:.*]] = arith.cmpi sgt, %[[VAL_70]], %[[VAL_26]] : index
! CHECK:  %[[VAL_72:.*]] = arith.select %[[VAL_71]], %[[VAL_70]], %[[VAL_26]] : index
! CHECK:  %[[VAL_73:.*]] = hlfir.designate %[[VAL_14]]#0 (%[[VAL_25]]:%[[VAL_27]]#1:%[[VAL_25]]) substr %[[VAL_67]], %[[VAL_68]]  shape %[[VAL_33]] typeparams %[[VAL_72]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>

  print *, scalar(n:m)
! CHECK:  %[[VAL_79:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_80:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_81:.*]] = fir.convert %[[VAL_79]] : (i64) -> index
! CHECK:  %[[VAL_82:.*]] = fir.convert %[[VAL_80]] : (i64) -> index
! CHECK:  %[[VAL_83:.*]] = arith.subi %[[VAL_82]], %[[VAL_81]] : index
! CHECK:  %[[VAL_84:.*]] = arith.addi %[[VAL_83]], %[[VAL_25]] : index
! CHECK:  %[[VAL_85:.*]] = arith.cmpi sgt, %[[VAL_84]], %[[VAL_26]] : index
! CHECK:  %[[VAL_86:.*]] = arith.select %[[VAL_85]], %[[VAL_84]], %[[VAL_26]] : index
! CHECK:  %[[VAL_87:.*]] = hlfir.designate %[[VAL_19]]#0  substr %[[VAL_81]], %[[VAL_82]]  typeparams %[[VAL_86]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
end subroutine

subroutine test_static_substring(i, j)
  integer(8) :: i, j
  print *, "hello"(i:j)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_static_substring(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}}i"
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}j"
! CHECK:  %[[VAL_10:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = ".stringlit"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:  %[[VAL_12:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:  %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_17:.*]] = arith.subi %[[VAL_15]], %[[VAL_14]] : index
! CHECK:  %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_16]] : index
! CHECK:  %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_19]] : index
! CHECK:  %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_18]], %[[VAL_19]] : index
! CHECK:  %[[VAL_22:.*]] = hlfir.designate %[[VAL_11]]#0  substr %[[VAL_14]], %[[VAL_15]]  typeparams %[[VAL_21]] : (!fir.ref<!fir.char<1,5>>, index, index, index) -> !fir.boxchar<1>
