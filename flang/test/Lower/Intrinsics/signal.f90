! test lowering of the SIGNAL intrinsic subroutine
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module m
contains
! CHECK-LABEL:   func.func @handler(
! CHECK-SAME:                       %[[VAL_0:.*]]: i32 {fir.bindc_name = "signum"}) attributes {fir.bindc_name = "handler"} {
  subroutine handler(signum) bind(C)
    use iso_c_binding, only: c_int
    integer(c_int), value :: signum
  end subroutine

! CHECK-LABEL:   func.func @_QMmPsetup_signals(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "optional_status", fir.optional}) {
  subroutine setup_signals(optional_status)
    ! not portable accross systems
    integer, parameter :: SIGFPE = 8
    integer, parameter :: SIGUSR1 = 10
    integer, parameter :: SIGUSR2 = 12
    integer, parameter :: SIGPIPE = 13
    integer, parameter :: SIG_IGN = 1
    integer :: stat = 0
    integer, optional, intent(out) :: optional_status

! CHECK:           %[[VAL_1:.*]] = fir.alloca i32
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_out, optional>, uniq_name = "_QMmFsetup_signalsEoptional_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMmFsetup_signalsEstat"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    call signal(SIGFPE, handler)
! CHECK:           %[[VAL_15:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_16:.*]] = fir.address_of(@handler) : (i32) -> ()
! CHECK:           %[[VAL_17:.*]] = fir.emboxproc %[[VAL_16]] : ((i32) -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_19:.*]] = fir.box_addr %[[VAL_17]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_20:.*]] = fir.call @_FortranASignal(%[[VAL_18]], %[[VAL_19]]) fastmath<contract> : (i64, () -> ()) -> i64

    call signal(SIGUSR1, handler, stat)
! CHECK:           %[[VAL_21:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_22:.*]] = fir.address_of(@handler) : (i32) -> ()
! CHECK:           %[[VAL_23:.*]] = fir.emboxproc %[[VAL_22]] : ((i32) -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.box_addr %[[VAL_23]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_26:.*]] = fir.call @_FortranASignal(%[[VAL_24]], %[[VAL_25]]) fastmath<contract> : (i64, () -> ()) -> i64
! CHECK:           %[[VAL_27:.*]] = fir.is_present %[[VAL_14]]#1 : (!fir.ref<i32>) -> i1
! CHECK:           fir.if %[[VAL_27]] {
! CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_26]] : (i64) -> i32
! CHECK:             fir.store %[[VAL_28]] to %[[VAL_14]]#1 : !fir.ref<i32>
! CHECK:           }

    call signal(SIGUSR2, SIG_IGN, stat)
! CHECK:           %[[VAL_29:.*]] = arith.constant 12 : i32
! CHECK:           %[[VAL_30:.*]] = arith.constant 1 : i32
! CHECK:           fir.store %[[VAL_30]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> !fir.llvm_ptr<() -> ()>
! CHECK:           %[[VAL_34:.*]] = fir.call @_FortranASignal(%[[VAL_31]], %[[VAL_33]]) fastmath<contract> : (i64, !fir.llvm_ptr<() -> ()>) -> i64
! CHECK:           %[[VAL_35:.*]] = fir.is_present %[[VAL_14]]#1 : (!fir.ref<i32>) -> i1
! CHECK:           fir.if %[[VAL_35]] {
! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (i64) -> i32
! CHECK:             fir.store %[[VAL_36]] to %[[VAL_14]]#1 : !fir.ref<i32>
! CHECK:           }

    call signal(SIGPIPE, handler, optional_status)
! CHECK:           %[[VAL_37:.*]] = arith.constant 13 : i32
! CHECK:           %[[VAL_38:.*]] = fir.address_of(@handler) : (i32) -> ()
! CHECK:           %[[VAL_39:.*]] = fir.emboxproc %[[VAL_38]] : ((i32) -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_37]] : (i32) -> i64
! CHECK:           %[[VAL_41:.*]] = fir.box_addr %[[VAL_39]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_42:.*]] = fir.call @_FortranASignal(%[[VAL_40]], %[[VAL_41]]) fastmath<contract> : (i64, () -> ()) -> i64
! CHECK:           %[[VAL_43:.*]] = fir.is_present %[[VAL_2]]#1 : (!fir.ref<i32>) -> i1
! CHECK:           fir.if %[[VAL_43]] {
! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_42]] : (i64) -> i32
! CHECK:             fir.store %[[VAL_44]] to %[[VAL_2]]#1 : !fir.ref<i32>
! CHECK:           }
  end subroutine
end module
