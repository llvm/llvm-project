! RUN: bbc -polymorphic-type -emit-fir -hlfir=false %s -o - | FileCheck %s

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
! CHECK: %[[CONV:.*]] = fir.convert %[[I]] : (!fir.ref<i32>) -> !fir.ref<none>
! CHECK: fir.call @_QPassumed(%[[CONV]]) {{.*}}: (!fir.ref<none>) -> ()

  subroutine call_assumed_r()
    integer, target :: i(10)
    call assumed_r(i)
  end subroutine

! CHECK-LABEL: func.func @_QMassumed_type_testPcall_assumed_r() {
! CHECK: %[[I:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "i", fir.target, uniq_name = "_QMassumed_type_testFcall_assumed_rEi"}
! CHECK: %[[CONV:.*]] = fir.convert %[[I]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK: fir.call @_QPassumed_r(%[[CONV]]) {{.*}} : (!fir.ref<!fir.array<?xnone>>) -> ()

  subroutine assumed_type_optional_to_intrinsic(a)
    type(*), optional :: a(:)
    if (present(a)) print*, 'present'
  end subroutine

! CHECK-LABEL: func.func @_QMassumed_type_testPassumed_type_optional_to_intrinsic(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "a", fir.optional}) {
! CHECK: %{{.*}} = fir.is_present %[[ARG0]] : (!fir.box<!fir.array<?xnone>>) -> i1

  subroutine assumed_type_lbound(a)
    type(*), optional :: a(:,:)
    print*,lbound(a,dim=1)
  end subroutine

! CHECK-LABEL: func.func @_QMassumed_type_testPassumed_type_lbound(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x?xnone>> {fir.bindc_name = "a", fir.optional}) {
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[C1]]) {{.*}} : (!fir.ref<i8>, i32) -> i1

end module
