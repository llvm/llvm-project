! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPiany_test_1(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi8>>{{.*}}) -> i8 {
integer(1) function iany_test_1(a)
integer(1) :: a(:)
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[arg0]]
iany_test_1 = iany(a)
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.box<!fir.array<?xi8>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAIAny1(%[[A_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i8
end function

! CHECK-LABEL: func.func @_QPiany_test_2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi16>>{{.*}}) -> i16 {
integer(2) function iany_test_2(a)
integer(2) :: a(:)
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[arg0]]
iany_test_2 = iany(a)
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.box<!fir.array<?xi16>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAIAny2(%[[A_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i16
end function

! CHECK-LABEL: func.func @_QPiany_test_4(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32 {
integer function iany_test_4(a)
integer :: a(:)
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[arg0]]
iany_test_4 = iany(a)
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAIAny4(%[[A_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end function

! CHECK-LABEL: func.func @_QPiany_test_8(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi64>>{{.*}}) -> i64 {
integer(8) function iany_test_8(a)
integer(8) :: a(:)
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[arg0]]
iany_test_8 = iany(a)
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAIAny8(%[[A_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i64
end function

! CHECK-LABEL: func.func @_QPiany_test_16(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi128>>{{.*}}) -> i128 {
integer(16) function iany_test_16(a)
integer(16) :: a(:)
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[arg0]]
iany_test_16 = iany(a)
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.box<!fir.array<?xi128>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAIAny16(%[[A_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i128
end function

! CHECK-LABEL: func.func @_QPiany_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine iany_test2(a,r)
integer :: a(:,:)
integer :: r(:)
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[R:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
r = iany(a,dim=2)
! CHECK: fir.call @_FortranAIAnyDim(%{{.*}}, %[[A_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>) -> ()
end subroutine

! CHECK-LABEL: func.func @_QPiany_test_optional(
! CHECK-SAME:  %[[MASK_ARG:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}}, %[[X_ARG:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
integer function iany_test_optional(mask, x)
integer :: x(:)
logical, optional :: mask(:)
! CHECK: %[[MASK:.*]]:2 = hlfir.declare %[[MASK_ARG]]
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ARG]]
iany_test_optional = iany(x, mask=mask)
! CHECK: %[[X_NONE:.*]] = fir.convert %[[X]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK: %[[MASK_NONE:.*]] = fir.convert %[[MASK]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAIAny4(%[[X_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %[[MASK_NONE]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end function

! CHECK-LABEL: func.func @_QPiany_test_optional_2(
! CHECK-SAME:  %[[MASK_ARG:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>{{.*}})
integer function iany_test_optional_2(mask, x)
integer :: x(:)
logical, pointer :: mask(:)
! CHECK: %[[MASK:.*]]:2 = hlfir.declare %[[MASK_ARG]]
iany_test_optional_2 = iany(x, mask=mask)
! CHECK: %[[LOAD_MASK:.*]] = fir.load %[[MASK]]#0
! CHECK: %[[MASK_ADDR:.*]] = fir.box_addr %[[LOAD_MASK]]
! CHECK: %{{.*}} = arith.cmpi ne, %{{.*}}, %{{.*}} : i64
! CHECK: %[[SEL_MASK:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>
! CHECK: %[[MASK_NONE:.*]] = fir.convert %[[SEL_MASK]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAIAny4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[MASK_NONE]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end function

! CHECK-LABEL: func.func @_QPiany_test_optional_3(
! CHECK-SAME:  %[[MASK_ARG:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>
integer function iany_test_optional_3(mask, x)
integer :: x(:)
logical, optional :: mask(10)
! CHECK: %[[MASK:.*]]:2 = hlfir.declare %[[MASK_ARG]]
iany_test_optional_3 = iany(x, mask=mask)
! CHECK: %[[PRESENT:.*]] = fir.is_present %[[MASK]]#0
! CHECK: %[[EMBOX:.*]] = fir.embox %[[MASK]]#0
! CHECK: %[[SEL_MASK:.*]] = arith.select %[[PRESENT]], %[[EMBOX]], %{{.*}} : !fir.box<!fir.array<10x!fir.logical<4>>>
! CHECK: %[[MASK_NONE:.*]] = fir.convert %[[SEL_MASK]] : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAIAny4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[MASK_NONE]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end function

! CHECK-LABEL: func.func @_QPiany_test_optional_4(
integer function iany_test_optional_4(x, use_mask)
! Test that local allocatable tracked in local variables
! are dealt as optional argument correctly.
integer :: x(:)
logical :: use_mask
logical, allocatable :: mask(:)
! CHECK: %[[MASK:.*]]:2 = hlfir.declare %{{.*}}mask"}
if (use_mask) then
  allocate(mask(size(x, 1)))
  call set_mask(mask)
  ! CHECK: fir.call @_QPset_mask
end if
iany_test_optional_4 = iany(x, mask=mask)
! CHECK: %[[LOAD_MASK:.*]] = fir.load %[[MASK]]#0
! CHECK: %[[MASK_ADDR:.*]] = fir.box_addr %[[LOAD_MASK]]
! CHECK: %[[CMPI:.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : i64
! CHECK: %[[SEL_MASK:.*]] = arith.select %[[CMPI]], %{{.*}}, %{{.*}} : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK: %[[MASK_NONE:.*]] = fir.convert %[[SEL_MASK]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAIAny4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[MASK_NONE]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
end function
