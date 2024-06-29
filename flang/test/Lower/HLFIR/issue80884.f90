! Test lowering of pointer remapping with component ref in the RHS.
! RUN: bbc -emit-hlfir -o - %s -I nw | FileCheck %s

subroutine issue80884(p, targ)
  type t0
    real :: array(10, 10)
  end type
  type, extends(t0) :: t
  end type
  type(t), target :: targ
  real, pointer :: p(:)
  p(1:100) => targ%array
end subroutine
! CHECK-LABEL:   func.func @_QPissue80884(
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFissue80884Ep"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFissue80884Etarg"} : (!fir.ref<!fir.type<_QFissue80884Tt{t0:!fir.type<_QFissue80884Tt0{array:!fir.array<10x10xf32>}>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QFissue80884Tt{t0:!fir.type<_QFissue80884Tt0{array:!fir.array<10x10xf32>}>}>>, !fir.ref<!fir.type<_QFissue80884Tt{t0:!fir.type<_QFissue80884Tt0{array:!fir.array<10x10xf32>}>}>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_5:.*]] = arith.constant 100 : i64
! CHECK:           %[[VAL_6:.*]] = hlfir.designate %[[VAL_3]]#0{"t0"}   : (!fir.ref<!fir.type<_QFissue80884Tt{t0:!fir.type<_QFissue80884Tt0{array:!fir.array<10x10xf32>}>}>>) -> !fir.ref<!fir.type<_QFissue80884Tt0{array:!fir.array<10x10xf32>}>>
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_6]]{"array"}   shape %[[VAL_9]] : (!fir.ref<!fir.type<_QFissue80884Tt0{array:!fir.array<10x10xf32>}>>, !fir.shape<2>) -> !fir.ref<!fir.array<10x10xf32>>
! CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:           %[[VAL_14:.*]] = arith.subi %[[VAL_13]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_11]] : index
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[VAL_17:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_15]] : (i64, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_16]](%[[VAL_17]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_2]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
