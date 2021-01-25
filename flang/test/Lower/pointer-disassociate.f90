! Test lowering of pointer disassociation
! RUN: bbc -emit-fir %s -o - | FileCheck %s


! -----------------------------------------------------------------------------
!     Test p => NULL()
! -----------------------------------------------------------------------------


! CHECK-LABEL: func @_QPtest_scalar(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>)
subroutine test_scalar(p)
  real, pointer :: p
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[null]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  p => NULL()
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_char(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
subroutine test_scalar_char(p)
  character(:), pointer :: p
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[box:.*]] = fir.embox %[[null]] typeparams %c0{{.*}} : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  p => NULL()
end subroutine

! CHECK-LABEL: func @_QPtest_array(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
subroutine test_array(p)
  real, pointer :: p(:)
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape %c0{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[null]](%[[shape]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => NULL()
end subroutine

! Test p(lb, ub) => NULL() which is none sens but is not illegal.
! CHECK-LABEL: func @_QPtest_array_remap(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
subroutine test_array_remap(p)
  real, pointer :: p(:)
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape %c0{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[null]](%[[shape]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p(10:20) => NULL()
end subroutine


! TODO: p => NULL(MOLD). Requires array function/intrinsic lowering work. 

