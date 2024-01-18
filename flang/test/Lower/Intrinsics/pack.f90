! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: pack_test
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-SAME: %[[arg2:[^:]+]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[arg3:[^:]+]]: !fir.box<!fir.array<?xi32>>
subroutine pack_test(a,m,v,r)
    integer :: a(:)
    logical :: m(:)
    integer :: v(:)
    integer :: r(:)
  ! CHECK:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK:  %[[a5:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:  %[[a6:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK:  %[[a7:.*]] = fir.convert %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK:  %[[a8:.*]] = fir.convert %[[arg2]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
    r = pack(a,m,v)
  ! CHECK: %{{.*}} = fir.call @_FortranAPack(%[[a5]], %[[a6]], %[[a7]], %[[a8]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:  %[[a11:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK:  %[[a13:.*]] = fir.box_addr %[[a11]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK:  fir.freemem %[[a13]]
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_pack_optional(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  subroutine test_pack_optional(vector, array, mask)
    integer, pointer :: vector(:)
    integer :: array(:, :)
    logical :: mask(:, :)
    print *, pack(array, mask, vector)
  ! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
  ! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
  ! CHECK:  %[[VAL_12:.*]] = arith.constant 0 : i64
  ! CHECK:  %[[VAL_13:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_12]] : i64
  ! CHECK:  %[[VAL_14:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  %[[VAL_15:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[VAL_16:.*]] = arith.select %[[VAL_13]], %[[VAL_14]], %[[VAL_15]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[VAL_26:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<none>
  ! CHECK:  fir.call @_FortranAPack(%{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_26]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  end subroutine
