! Test that the selector is lowered as a box with the lower
! bounds inherited from the original variable. Otherwise,
! the bounds accessed inside the type guard block are going
! to be default 1-based.
!
! 11.1.11.2 Execution of the SELECT TYPE construct
! 5 Within the block following a TYPE IS type guard statement, the associating entity (19.5.5) is not polymorphic
! (7.3.2.3), has the type named in the type guard statement, and has the type parameter values of the selector.
! 8 The other attributes of the associating entity are described in 11.1.3.3.
! ...
! 11.1.3.3 Other attributes of associate names
! 1 Within an ASSOCIATE, CHANGE TEAM, or SELECT TYPE construct, each associating entity has the same
! rank as its associated selector. The lower bound of each dimension is the result of the intrinsic function LBOUND
! (16.9.109) applied to the corresponding dimension of selector. The upper bound of each dimension is one less
! than the sum of the lower bound and the extent.

! RUN: bbc -emit-hlfir -polymorphic-type -I nowhere -o - %s | FileCheck %s

subroutine test()
  type t
  end type t
  class(*), allocatable :: x(:)
  integer :: ub
!  allocate(x(-1:8))
  select type(x)
  type is (t)
     ub = ubound(x, 1)
  end select
end subroutine test
! CHECK-LABEL:   func.func @_QPtest() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "ub", uniq_name = "_QFtestEub"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtestEub"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?xnone>>> {bindc_name = "x", uniq_name = "_QFtestEx"}
! CHECK:           %[[VAL_3:.*]] = fir.zero_bits !fir.heap<!fir.array<?xnone>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_3]](%[[VAL_5]]) : (!fir.heap<!fir.array<?xnone>>, !fir.shape<1>) -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEx"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_10]] : (!fir.class<!fir.heap<!fir.array<?xnone>>>, index) -> (index, index, index)
! CHECK:           fir.select_type %[[VAL_8]] : !fir.class<!fir.heap<!fir.array<?xnone>>> [#fir.type_is<!fir.type<_QFtestTt>>, ^bb1, unit, ^bb2]
! CHECK:         ^bb1:
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_8]] : (!fir.class<!fir.heap<!fir.array<?xnone>>>) -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtestTt>>>>
! CHECK:           %[[VAL_13:.*]] = fir.shift %[[VAL_11]]#0 : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_13]]) {uniq_name = "_QFtestEx"} : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtestTt>>>>, !fir.shift<1>) -> (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtestTt>>>>, !fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtestTt>>>>)
! CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_14]]#0, %[[VAL_15]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtestTt>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]]#1 : (index) -> i64
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_11]]#0 : (index) -> i64
! CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_18]] : i64
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_19]], %[[VAL_20]] : i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> i32
! CHECK:           hlfir.assign %[[VAL_22]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
! CHECK:           cf.br ^bb2
! CHECK:         ^bb2:
! CHECK:           return
! CHECK:         }
