!RUN: bbc -emit-fir %s -o - 2>&1 | FileCheck %s

module m
  type t
    integer :: n = 0
   contains
    procedure :: tbp => f
  end type
 contains
  function f(this)
    class(t), pointer, intent(in) :: this
    integer, pointer :: f
    f => this%n
  end
end

subroutine test
  use m
  type(t), target :: xt
  class(t), pointer :: xp
  xp => xt
  xt%tbp() = 1
  xp%tbp() = 2
end

! CHECK-LABEL: func @_QPtest(
! CHECK:  %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK:  %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = ".result"}
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = ".result"}
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>
! CHECK:  %{{.*}} = fir.dummy_scope : !fir.dscope
! CHECK:  %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>> {bindc_name = "xp", uniq_name = "_QFtestExp"}
! CHECK:  %[[VAL_6:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QMmTt{n:i32}>>
! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_6]] : (!fir.ptr<!fir.type<_QMmTt{n:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>
! CHECK:  fir.store %[[VAL_7]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>
! CHECK:  %[[VAL_8:.*]] = fir.declare %[[VAL_5]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestExp"} : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>) -> !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>
! CHECK:  %[[VAL_9:.*]] = fir.alloca !fir.type<_QMmTt{n:i32}> {bindc_name = "xt", fir.target, uniq_name = "_QFtestExt"}
! CHECK:  %[[VAL_10:.*]] = fir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestExt"} : (!fir.ref<!fir.type<_QMmTt{n:i32}>>) -> !fir.ref<!fir.type<_QMmTt{n:i32}>>
! CHECK:  %[[VAL_11:.*]] = fir.address_of(@_QQ_QMmTt.DerivedInit) : !fir.ref<!fir.type<_QMmTt{n:i32}>>
! CHECK:  fir.copy %[[VAL_11]] to %[[VAL_10]] no_overlap : !fir.ref<!fir.type<_QMmTt{n:i32}>>, !fir.ref<!fir.type<_QMmTt{n:i32}>>
! CHECK:  %[[VAL_12:.*]] = fir.embox %[[VAL_10]] : (!fir.ref<!fir.type<_QMmTt{n:i32}>>) -> !fir.box<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>
! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>) -> !fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>
! CHECK:  fir.store %[[VAL_13]] to %[[VAL_8]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>
! CHECK:  %[[VAL_14:.*]] = fir.embox %[[VAL_10]] : (!fir.ref<!fir.type<_QMmTt{n:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>
! CHECK:  fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>
! CHECK:  %[[VAL_15:.*]] = fir.call @_QMmPf(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>) -> !fir.box<!fir.ptr<i32>>
! CHECK:  fir.save_result %[[VAL_15]] to %[[VAL_2]] : !fir.box<!fir.ptr<i32>>, !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_16:.*]] = fir.declare %[[VAL_2]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_18:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  fir.store %[[C1_I32]] to %[[VAL_18]] : !fir.ptr<i32>
! CHECK:  %[[VAL_19:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>
! CHECK:  %[[VAL_20:.*]] = fir.rebox %[[VAL_19]] : (!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>) -> !fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>
! CHECK:  fir.store %[[VAL_20]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>
! CHECK:  %[[VAL_21:.*]] = fir.dispatch "tbp"(%[[VAL_19]] : !fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>) (%[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmTt{n:i32}>>>>) -> !fir.box<!fir.ptr<i32>> {pass_arg_pos = 0 : i32}
! CHECK:  fir.save_result %[[VAL_21]] to %[[VAL_0]] : !fir.box<!fir.ptr<i32>>, !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_22:.*]] = fir.declare %[[VAL_0]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_24:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  fir.store %[[C2_I32]] to %[[VAL_24]] : !fir.ptr<i32>
