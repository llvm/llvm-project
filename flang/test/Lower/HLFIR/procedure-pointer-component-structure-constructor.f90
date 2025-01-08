! Test passing
!  1. NULL(),
!  2. procedure,
!  3. procedure pointer,
!  4. reference to a function that returns a procedure pointer.
! to a derived type structure constructor.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

  MODULE M
    TYPE :: DT
      PROCEDURE(Fun), POINTER, NOPASS :: pp1
    END TYPE

    CONTAINS

    INTEGER FUNCTION Fun(Arg)
    INTEGER :: Arg
      Fun = Arg
    END FUNCTION

  END MODULE

  PROGRAM MAIN
  USE M
  IMPLICIT NONE
  TYPE (DT), PARAMETER :: v1 = DT(NULL())
  TYPE (DT) :: v2
  PROCEDURE(FUN), POINTER :: pp2
  v2 = DT(fun)
  v2 = DT(pp2)
  v2 = DT(bar())
  CONTAINS
    FUNCTION BAR() RESULT(res)
      PROCEDURE(FUN), POINTER :: res
    END
  END

! CHECK-LABEL:  func.func @_QQmain() attributes {fir.bindc_name = "main"} {
! CHECK:    %[[VAL_0:.*]] = fir.alloca !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:    %[[VAL_1:.*]] = fir.alloca !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:    %[[VAL_2:.*]] = fir.alloca !fir.boxproc<(!fir.ref<i32>) -> i32> {bindc_name = "pp2", uniq_name = "_QFEpp2"}
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEpp2"} : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> i32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> i32>>, !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> i32>>)
! CHECK:    %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>>) -> (!fir.ref<!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>>, !fir.ref<!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>>)
! CHECK:    %[[VAL_23:.*]] = hlfir.designate %[[VAL_17]]#0{"pp1"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> i32>>
! CHECK:    %[[VAL_24:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> i32>>
! CHECK:    fir.store %[[VAL_24]] to %[[VAL_23]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> i32>>
! CHECK:    %[[VAL_25:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>>) -> (!fir.ref<!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>>, !fir.ref<!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>>)
! CHECK:    %[[VAL_31:.*]] = hlfir.designate %[[VAL_25]]#0{"pp1"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> i32>>
! CHECK:    %[[VAL_32:.*]] = fir.call @_QFPbar() fastmath<contract> : () -> !fir.boxproc<(!fir.ref<i32>) -> i32>
! CHECK:    fir.store %[[VAL_32]] to %[[VAL_31]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> i32>>
! CHECK:    return
! CHECK:  }

! CHECK-LABEL:  fir.global internal @_QFECv1 constant : !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}> {
! CHECK:    %[[VAL_0:.*]] = fir.undefined !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:    %[[VAL_1:.*]] = fir.field_index pp1, !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:    %[[VAL_2:.*]] = fir.zero_bits (!fir.ref<i32>) -> i32
! CHECK:    %[[VAL_3:.*]] = fir.emboxproc %[[VAL_2]] : ((!fir.ref<i32>) -> i32) -> !fir.boxproc<(!fir.ref<i32>) -> i32>
! CHECK:    %[[VAL_4:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_3]], ["pp1", !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>] : (!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>, !fir.boxproc<(!fir.ref<i32>) -> i32>) -> !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:    fir.has_value %[[VAL_4]] : !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:  }

! CHECK-LABEL:  fir.global internal @_QQro._QMmTdt.0 constant : !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}> {
! CHECK:    %[[VAL_0:.*]] = fir.undefined !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:    %[[VAL_1:.*]] = fir.field_index pp1, !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:    %[[VAL_2:.*]] = fir.address_of(@_QMmPfun) : (!fir.ref<i32>) -> i32
! CHECK:    %[[VAL_3:.*]] = fir.emboxproc %[[VAL_2]] : ((!fir.ref<i32>) -> i32) -> !fir.boxproc<() -> ()>
! CHECK:    %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<i32>) -> i32>
! CHECK:    %[[VAL_5:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_4]], ["pp1", !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>] : (!fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>, !fir.boxproc<(!fir.ref<i32>) -> i32>) -> !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:    fir.has_value %[[VAL_5]] : !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}>
! CHECK:  }
