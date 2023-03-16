! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

module assumed_type_test

  interface
    subroutine assumed(a)
      type(*), intent(in), target :: a
    end subroutine
  end interface

contains

  subroutine call_assmued()
    integer, target :: i
    call assumed(i)
  end subroutine

! CHECK-LABEL: func.func @_QMassumed_type_testPcall_assmued() {
! CHECK: %[[I:.*]] = fir.alloca i32 {bindc_name = "i", fir.target, uniq_name = "_QMassumed_type_testFcall_assmuedEi"}
! CHECK: %[[BOX_NONE:.*]] = fir.embox %[[I]] : (!fir.ref<i32>) -> !fir.box<none>
! CHECK: fir.call @_QPassumed(%[[BOX_NONE]]) fastmath<contract> : (!fir.box<none>) -> ()

end module
