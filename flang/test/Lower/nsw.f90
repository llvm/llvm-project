! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc -emit-fir -fwrapv %s -o - | FileCheck %s --check-prefix=NO-NSW

! NO-NSW-NOT: overflow<nsw>

subroutine subscript(a, i, j, k)
  integer :: a(:,:,:), i, j, k
  a(i+1, j-2, k*3) = 5
end subroutine
! CHECK-LABEL:   func.func @_QPsubscript(
! CHECK:           %[[VAL_4:.*]] = arith.constant 3 : i32
! CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_9:.*]] = fir.declare %{{.*}}a"} : (!fir.box<!fir.array<?x?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?x?xi32>>
! CHECK:           %[[VAL_10:.*]] = fir.rebox %[[VAL_9]] : (!fir.box<!fir.array<?x?x?xi32>>) -> !fir.box<!fir.array<?x?x?xi32>>
! CHECK:           %[[VAL_11:.*]] = fir.declare %{{.*}}i"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_12:.*]] = fir.declare %{{.*}}j"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_13:.*]] = fir.declare %{{.*}}k"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_6]] overflow<nsw> : i32
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_5]] overflow<nsw> : i32
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:           %[[VAL_21:.*]] = arith.muli %[[VAL_20]], %[[VAL_4]] overflow<nsw> : i32
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:           %[[VAL_23:.*]] = fir.array_coor %[[VAL_10]] %[[VAL_16]], %[[VAL_19]], %[[VAL_22]] :

! Test that nsw is never added to arith ops
! on vector subscripts.
subroutine vector_subscript_as_value(x, y, z)
  integer :: x(100)
  integer(8) :: y(20), z(20)
  call bar(x(y+z))
end subroutine
! CHECK-LABEL:   func.func @_QPvector_subscript_as_value(
! CHECK-NOT: overflow<nsw>
! CHECK:           return

subroutine vector_subscript_lhs(x, vector1, vector2)
  integer(8) :: vector1(10), vector2(10)
  real :: x(:)
  x(vector1+vector2) = 42.
end subroutine
! CHECK-LABEL:   func.func @_QPvector_subscript_lhs(
! CHECK-NOT: overflow<nsw>
! CHECK:           return

! Test that nsw is never added to arith ops
! on arguments of bitwise comparison intrinsics.
subroutine bitwise_comparison(a, b)
  integer :: a, b
  print *, bge(a+b, a-b)
  print *, bgt(a+b, a-b)
  print *, ble(a+b, a-b)
  print *, blt(a+b, a-b)
end subroutine
! CHECK-LABEL:   func.func @_QPbitwise_comparison(
! CHECK-NOT: overflow<nsw>
! CHECK:           return
