! Test passing
!  1. NULL(),
!  2. procedure,
!  3. procedure pointer, (pending)
!  4. reference to a function that returns a procedure pointer (pending)
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
  v2 = DT(fun)
  END

! CDHECK-LABEL:  fir.global internal @_QFECv1 constant : !fir.type<_QMmTdt{pp1:!fir.boxproc<(!fir.ref<i32>) -> i32>}> {
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
