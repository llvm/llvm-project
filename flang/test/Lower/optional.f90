! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test OPTIONAL lowering on caller/callee and PRESENT intrinsic.
module opt
contains

! Test simple scalar optional
! CHECK-LABEL: func @_QMoptPintrinsic_scalar
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<f32> {fir.optional})
subroutine intrinsic_scalar(x)
  implicit none
  real, optional :: x
  ! CHECK: fir.is_present %[[arg0]] : (!fir.ref<f32>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: @_QMoptPcall_intrinsic_scalar()
subroutine call_intrinsic_scalar()
  implicit none
  ! CHECK: %[[x:.*]] = fir.alloca f32
  real :: x
  ! CHECK: fir.call @_QMoptPintrinsic_scalar(%[[x]]) : (!fir.ref<f32>) -> ()
  call intrinsic_scalar(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.ref<f32>
  ! CHECK: fir.call @_QMoptPintrinsic_scalar(%[[absent]]) : (!fir.ref<f32>) -> ()
  call intrinsic_scalar()
end subroutine

! Test explicit shape array optional
! CHECK-LABEL: func @_QMoptPintrinsic_f77_array
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<!fir.array<100xf32>> {fir.optional})
subroutine intrinsic_f77_array(x)
  implicit none
  real, optional :: x(100)
  ! CHECK: fir.is_present %[[arg0]] : (!fir.ref<!fir.array<100xf32>>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: func @_QMoptPcall_intrinsic_f77_array()
subroutine call_intrinsic_f77_array()
  implicit none
  ! CHECK: %[[x:.*]] = fir.alloca !fir.array<100xf32>
  real :: x(100)
  ! CHECK: fir.call @_QMoptPintrinsic_f77_array(%[[x]]) : (!fir.ref<!fir.array<100xf32>>) -> ()
  call intrinsic_f77_array(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
  ! CHECK: fir.call @_QMoptPintrinsic_f77_array(%[[absent]]) : (!fir.ref<!fir.array<100xf32>>) -> ()
  call intrinsic_f77_array()
end subroutine

! Test optional character scalar
! CHECK-LABEL: func @_QMoptPcharacter_scalar
! CHECK-SAME: (%[[arg0:.*]]: !fir.boxchar<1> {fir.optional})
subroutine character_scalar(x)
  implicit none
  ! CHECK: %[[unboxed:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  character(10), optional :: x
  ! CHECK: fir.is_present %[[unboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
  print *, present(x)
end subroutine
! CHECK-LABEL: func @_QMoptPcall_character_scalar()
subroutine call_character_scalar()
  implicit none
  ! CHECK: %[[addr:.*]] = fir.alloca !fir.char<1,10>
  character(10) :: x
  ! CHECK: %[[addrCast:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[x:.*]] = fir.emboxchar %[[addrCast]], {{.*}}
  ! CHECK: fir.call @_QMoptPcharacter_scalar(%[[x]]) : (!fir.boxchar<1>) -> ()
  call character_scalar(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.boxchar<1>
  ! CHECK: fir.call @_QMoptPcharacter_scalar(%[[absent]]) : (!fir.boxchar<1>) -> ()
  call character_scalar()
end subroutine

! Test optional assumed shape
! CHECK-LABEL: func @_QMoptPassumed_shape
! CHECK-SAME: (%[[arg0:.*]]: !fir.box<!fir.array<?xf32>> {fir.optional})
subroutine assumed_shape(x)
  implicit none
  ! CHECK: %[[boxaddr:.*]] = fir.box_addr %[[arg0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  real, optional :: x(:)
  ! CHECK: fir.is_present %[[boxaddr]] : (!fir.ref<!fir.array<?xf32>>) -> i1
  print *, present(x)
end subroutine
! CHECK: func @_QMoptPcall_assumed_shape()
subroutine call_assumed_shape()
  implicit none
  ! CHECK: %[[addr:.*]] = fir.alloca !fir.array<100xf32>
  real :: x(100)
  ! CHECK: %[[embox:.*]] = fir.embox %[[addr]]
  ! CHECK: %[[x:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<100xf32>>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[x]]) : (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[absent]]) : (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape()
end subroutine

! Test optional allocatable
! CHECK: func @_QMoptPallocatable_array
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.optional})
subroutine allocatable_array(x)
  implicit none
  real, allocatable, optional :: x(:)
  ! CHECK: fir.is_present %[[arg0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> i1
  print *, present(x)
end subroutine
! CHECK: func @_QMoptPcall_allocatable_array()
subroutine call_allocatable_array()
  implicit none
  ! CHECK: %[[x:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  real, allocatable :: x(:)
  ! CHECK: fir.call @_QMoptPallocatable_array(%[[x]]) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  call allocatable_array(x)
  ! CHECK: %[[absent:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: fir.call @_QMoptPallocatable_array(%[[absent]]) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  call allocatable_array()
end subroutine

! CHECK: func @_QMoptPallocatable_to_assumed_optional_array
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
subroutine allocatable_to_assumed_optional_array(x)
  implicit none
  real, allocatable :: x(:)
  ! CHECK: %[[embox:.*]] = fir.embox %{{.*}}

  ! CHECK: %[[xboxload:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[xptr:.*]] = fir.box_addr %[[xboxload]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[xaddr:.*]] = fir.convert %[[xptr]] : (!fir.heap<!fir.array<?xf32>>) -> i64
  ! CHECK: %[[isAlloc:.*]] = cmpi ne, %[[xaddr]], %c0{{.*}} : i64
  ! CHECK: %[[absent:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[actual:.*]] = select %[[isAlloc]], %[[embox]], %[[absent]] : !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QMoptPassumed_shape(%[[actual]]) : (!fir.box<!fir.array<?xf32>>) -> ()
  call assumed_shape(x)
end subroutine

end module
