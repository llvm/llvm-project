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

subroutine loop_params(a,lb,ub,st)
  integer :: i, lb, ub, st
  integer :: a(lb:ub)
  do i = lb+1, ub-1, st*2
    a(i) = i
  end do
end subroutine
! CHECK-LABEL:   func.func @_QPloop_params(
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_9:.*]] = fir.declare %{{.*}}lb"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.declare %{{.*}}ub"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_12:.*]] = fir.declare %{{.*}}i"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK:           %[[VAL_13:.*]] = fir.declare %{{.*}}st"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_14]], %[[VAL_5]] overflow<nsw> : i32
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> index
! CHECK:           %[[VAL_27:.*]] = arith.subi %[[VAL_16]], %[[VAL_5]] overflow<nsw> : i32
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:           %[[VAL_30:.*]] = arith.muli %[[VAL_29]], %[[VAL_4]] overflow<nsw> : i32
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> index
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_26]] : (index) -> i32
! CHECK:           %[[VAL_33:.*]]:2 = fir.do_loop %[[VAL_34:.*]] = %[[VAL_26]] to %[[VAL_28]] step %[[VAL_31]] iter_args(%[[VAL_35:.*]] = %[[VAL_32]]) -> (index, i32) {

subroutine loop_params2(a,lb,ub,st)
  integer :: i, lb, ub, st
  integer :: a(lb:ub)
  real :: ii
  do ii = lb+1, ub-1, st*2
    i = ii
    a(i) = i
  end do
end subroutine
! CHECK-LABEL:   func.func @_QPloop_params2(
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_8:.*]] = fir.alloca index
! CHECK:           %[[VAL_9:.*]] = fir.alloca f32
! CHECK:           %[[VAL_11:.*]] = fir.declare %{{.*}}lb"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_12:.*]] = fir.declare %{{.*}}ub"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = fir.declare %{{.*}}i"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.declare %{{.*}}ii"} : (!fir.ref<f32>) -> !fir.ref<f32>
! CHECK:           %[[VAL_17:.*]] = fir.declare %{{.*}}st"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_18]], %[[VAL_5]] overflow<nsw> : i32
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> f32
! CHECK:           %[[VAL_31:.*]] = arith.subi %[[VAL_20]], %[[VAL_5]] overflow<nsw> : i32
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> f32
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_17]] : !fir.ref<i32>
! CHECK:           %[[VAL_34:.*]] = arith.muli %[[VAL_33]], %[[VAL_4]] overflow<nsw> : i32
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> f32
! CHECK:           fir.store %[[VAL_35]] to %[[VAL_9]] : !fir.ref<f32>
! CHECK:           %[[VAL_36:.*]] = arith.subf %[[VAL_32]], %[[VAL_30]] fastmath<contract> : f32
! CHECK:           %[[VAL_37:.*]] = arith.addf %[[VAL_36]], %[[VAL_35]] fastmath<contract> : f32
! CHECK:           %[[VAL_38:.*]] = arith.divf %[[VAL_37]], %[[VAL_35]] fastmath<contract> : f32
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (f32) -> index
! CHECK:           fir.store %[[VAL_39]] to %[[VAL_8]] : !fir.ref<index>
! CHECK:           fir.store %[[VAL_30]] to %[[VAL_16]] : !fir.ref<f32>
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_8]] : !fir.ref<index>
! CHECK:           %[[VAL_41:.*]] = arith.cmpi sgt, %[[VAL_40]], %[[VAL_6]] : index
! CHECK:           cf.cond_br %[[VAL_41]], ^bb2, ^bb3
! CHECK:         ^bb2:

subroutine loop_params3(a,lb,ub,st)
  integer :: i, lb, ub, st
  integer :: a(lb:ub)
  do concurrent (i=lb+1:ub-1:st*2)
    a(i) = i
  end do
end subroutine
! CHECK-LABEL:   func.func @_QPloop_params3(
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_9:.*]] = fir.declare %{{.*}}i"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = fir.declare %{{.*}}lb"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_12:.*]] = fir.declare %{{.*}}ub"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = fir.declare %{{.*}}i"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = fir.declare %{{.*}}st"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_16]], %[[VAL_5]] overflow<nsw> : i32
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
! CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_18]], %[[VAL_5]] overflow<nsw> : i32
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
! CHECK:           %[[VAL_32:.*]] = arith.muli %[[VAL_31]], %[[VAL_4]] overflow<nsw> : i32
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> index
! CHECK:           fir.do_loop %[[VAL_34:.*]] = %[[VAL_28]] to %[[VAL_30]] step %[[VAL_33]] unordered {
