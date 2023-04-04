! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

module assumed_type_test

  interface
    subroutine assumed(a)
      type(*), intent(in), target :: a
    end subroutine
  end interface

  interface
    subroutine assumed_r(a)
      type(*), intent(in), target :: a(*)
    end subroutine
  end interface

contains

  subroutine call_assumed()
    integer, target :: i
    call assumed(i)
  end subroutine

! CHECK-LABEL: func.func @_QMassumed_type_testPcall_assumed() {
! CHECK: %[[I:.*]] = fir.alloca i32 {bindc_name = "i", fir.target, uniq_name = "_QMassumed_type_testFcall_assumedEi"}
! CHECK: %[[BOX_NONE:.*]] = fir.embox %[[I]] : (!fir.ref<i32>) -> !fir.box<none>
! CHECK: fir.call @_QPassumed(%[[BOX_NONE]]) fastmath<contract> : (!fir.box<none>) -> ()

  subroutine call_assumed_r()
    integer, target :: i(10)
    call assumed_r(i)
  end subroutine

! CHECK-LABEL: func.func @_QMassumed_type_testPcall_assumed_r() {
! CHECK: %[[C10:.*]] = arith.constant 10 : index
! CHECK: %[[I:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "i", fir.target, uniq_name = "_QMassumed_type_testFcall_assumed_rEi"}
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
! CHECK: %[[BOX_NONE:.*]] = fir.embox %[[I]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xnone>>
! CHECK: %[[CONV:.*]] = fir.convert %[[BOX_NONE]] : (!fir.box<!fir.array<10xnone>>) -> !fir.box<!fir.array<?xnone>>
! CHECK: fir.call @_QPassumed_r(%[[CONV]]) {{.*}} : (!fir.box<!fir.array<?xnone>>) -> ()

  subroutine assumed_type_optional_to_intrinsic(a)
    type(*), optional :: a(:)
    if (present(a)) print*, 'present'
  end subroutine

! CHECK-LABEL: func.func @_QMassumed_type_testPassumed_type_optional_to_intrinsic(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "a", fir.optional}) {
! CHECK: %{{.*}} = fir.is_present %[[ARG0]] : (!fir.box<!fir.array<?xnone>>) -> i1

end module
