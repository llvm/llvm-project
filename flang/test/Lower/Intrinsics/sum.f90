! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsum_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32
integer function sum_test(a)
integer :: a(:)
! CHECK: hlfir.declare %[[arg0]]
sum_test = sum(a)
! CHECK: hlfir.sum {{.*}} : (!fir.box<!fir.array<?xi32>>) -> i32
end function

! CHECK-LABEL: func @_QPsum_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>
subroutine sum_test2(a,r)
integer :: a(:,:)
integer :: r(:)
! CHECK-DAG:  %[[c2_i32:.*]] = arith.constant 2 : i32
! CHECK-DAG: %[[aDecl:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK-DAG: %[[rDecl:.*]]:2 = hlfir.declare %[[arg1]]
r = sum(a,dim=2)
! CHECK: hlfir.sum %[[aDecl]]#0 dim %[[c2_i32]] {{.*}} : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
end subroutine

! CHECK-LABEL: func @_QPsum_test3(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xcomplex<f32>>>{{.*}}) -> complex<f32>
complex function sum_test3(a)
complex :: a(:)
sum_test3 = sum(a)
! CHECK: hlfir.sum {{.*}} : (!fir.box<!fir.array<?xcomplex<f32>>>) -> complex<f32>
end function

! CHECK-LABEL: func @_QPsum_test4(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xcomplex<f80>>>{{.*}}) -> complex<f80>
complex(10) function sum_test4(x)
complex(10):: x(:)
sum_test4 = sum(x)
! CHECK: hlfir.sum {{.*}} : (!fir.box<!fir.array<?xcomplex<f80>>>) -> complex<f80>
end

! CHECK-LABEL: func @_QPsum_test_optional(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
integer function sum_test_optional(mask, x)
integer :: x(:)
logical, optional :: mask(:)
sum_test_optional = sum(x, mask=mask)
! CHECK-DAG:  %[[maskDecl:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}optional
! CHECK-DAG:  %[[xDecl:.*]]:2 = hlfir.declare %{{.*}} {{.*}}{uniq_name = "_QFsum_test_optionalEx"}
! CHECK:  %[[isPresent:.*]] = fir.is_present %[[maskDecl]]#0
! CHECK:  %[[absent:.*]] = fir.absent !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[opt:.*]] = arith.select %[[isPresent]], %[[maskDecl]]#1, %[[absent]]
! CHECK:  hlfir.sum %[[xDecl]]#0 mask %[[opt]] {{.*}} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
end function

! CHECK-LABEL: func @_QPsum_test_optional_2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
integer function sum_test_optional_2(mask, x)
integer :: x(:)
logical, pointer :: mask(:)
sum_test_optional = sum(x, mask=mask)
! CHECK:  %[[maskDecl:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:  %[[VAL_4:.*]] = fir.load %[[maskDecl]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>) -> !fir.ptr<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ptr<!fir.array<?x!fir.logical<4>>>) -> i64
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:  %[[VAL_9:.*]] = fir.load %[[maskDecl]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[VAL_10:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>
! CHECK:  %[[VAL_11:.*]] = arith.select %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>
! CHECK:  hlfir.sum {{.*}} mask %[[VAL_11]] {{.*}} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>) -> i32
end function

! CHECK-LABEL: func @_QPsum_test_optional_3(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>
integer function sum_test_optional_3(mask, x)
integer :: x(:)
logical, optional :: mask(10)
sum_test_optional = sum(x, mask=mask)
! CHECK:  %[[maskDecl:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:  %[[VAL_5:.*]] = fir.is_present %[[maskDecl]]#0 : (!fir.ref<!fir.array<10x!fir.logical<4>>>) -> i1
! CHECK:  %[[VAL_6:.*]] = fir.shape
! CHECK:  %[[VAL_7:.*]] = fir.embox %[[maskDecl]]#0(%[[VAL_6]])
! CHECK:  %[[VAL_8:.*]] = fir.absent !fir.box<!fir.array<10x!fir.logical<4>>>
! CHECK:  %[[VAL_9:.*]] = arith.select %[[VAL_5]], %[[VAL_7]], %[[VAL_8]] : !fir.box<!fir.array<10x!fir.logical<4>>>
! CHECK:  hlfir.sum {{.*}} mask %[[VAL_9]] {{.*}} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<10x!fir.logical<4>>>) -> i32
end function

! CHECK-LABEL: func @_QPsum_test_optional_4(
integer function sum_test_optional_4(x, use_mask)
! Test that local allocatable tracked in local variables
! are dealt as optional argument correctly.
integer :: x(:)
logical :: use_mask
logical, allocatable :: mask(:)
if (use_mask) then
  allocate(mask(size(x, 1)))
  call set_mask(mask)
  ! CHECK: fir.call @_QPset_mask
end if
sum_test_optional = sum(x, mask=mask)
! CHECK:  %[[VAL_20:.*]] = fir.load %[[maskBox:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[VAL_21:.*]] = fir.box_addr %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (!fir.heap<!fir.array<?x!fir.logical<4>>>) -> i64
! CHECK:  %[[VAL_23:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_24:.*]] = arith.cmpi ne, %[[VAL_22]], %[[VAL_23]] : i64
! CHECK:  %[[VAL_25:.*]] = fir.load %[[maskBox]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[VAL_26:.*]] = fir.absent !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK:  %[[VAL_27:.*]] = arith.select %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK:  hlfir.sum {{.*}} mask %[[VAL_27]] {{.*}} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> i32
end function
