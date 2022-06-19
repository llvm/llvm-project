! Test lowering of nullify-statement
! RUN: bbc -emit-fir %s -o - | FileCheck %s


! -----------------------------------------------------------------------------
!     Test NULLIFY(p)
! -----------------------------------------------------------------------------


! CHECK-LABEL: func @_QPtest_scalar(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{.*}})
subroutine test_scalar(p)
  real, pointer :: p
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[null]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  nullify(p)
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_char(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{.*}})
subroutine test_scalar_char(p)
  character(:), pointer :: p
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[box:.*]] = fir.embox %[[null]] typeparams %c0{{.*}} : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  nullify(p)
end subroutine

! CHECK-LABEL: func @_QPtest_array(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}})
subroutine test_array(p)
  real, pointer :: p(:)
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape %c0{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[null]](%[[shape]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  nullify(p)
end subroutine

! CHECK-LABEL: func @_QPtest_list(
! CHECK-SAME: %[[p1:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{.*}}, %[[p2:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}})
subroutine test_list(p1, p2)
  real, pointer :: p1, p2(:)
  ! CHECK: fir.zero_bits !fir.ptr<f32>
  ! CHECK: fir.store %{{.*}} to %[[p1]] : !fir.ref<!fir.box<!fir.ptr<f32>>>

  ! CHECK: fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: fir.store %{{.*}} to %[[p2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  nullify(p1, p2)
end subroutine
