! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module protected_target_lowering
  type t
    integer, pointer, protected_target :: c
  end type
contains
  function copy_pointer(p) result(q)
    integer, pointer, protected_target, intent(in) :: p
    integer, pointer, protected_target :: q
    q => p
  end
  subroutine allocate_pointer(p)
    integer, pointer, protected_target :: p
    allocate(p, source=1)
  end
  subroutine component(p, a)
    integer, pointer, protected_target, intent(in) :: p
    type(t), intent(out) :: a
    a = t(p)
  end
end

! CHECK-LABEL: func.func @_QMprotected_target_loweringPcopy_pointer(
! CHECK-SAME: !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-SAME: -> !fir.box<!fir.ptr<i32>>
! CHECK-LABEL: func.func @_QMprotected_target_loweringPallocate_pointer(
! CHECK-SAME: !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: fir.call @_FortranAPointerAllocateSource(
! CHECK-LABEL: func.func @_QMprotected_target_loweringPcomponent(
! CHECK-SAME: !fir.ref<!fir.box<!fir.ptr<i32>>>
