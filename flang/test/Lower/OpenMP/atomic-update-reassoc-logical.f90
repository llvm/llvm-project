!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

subroutine f00(x, y, z)
  implicit none
  logical :: x, y, z

  !$omp atomic update
  x = x .and. y .and. z
end

!CHECK-LABEL: func.func @_QPf00
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[Z:[0-9]+]]:2 = hlfir.declare %arg2
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<!fir.logical<4>>
!CHECK: %[[LOAD_Z:[0-9]+]] = fir.load %[[Z]]#0 : !fir.ref<!fir.logical<4>>
!CHECK: %[[CVT_Y:[0-9]+]] = fir.convert %[[LOAD_Y]] : (!fir.logical<4>) -> i1
!CHECK: %[[CVT_Z:[0-9]+]] = fir.convert %[[LOAD_Z]] : (!fir.logical<4>) -> i1
!CHECK: %[[AND_YZ:[0-9]+]] = arith.andi %[[CVT_Y]], %[[CVT_Z]] : i1
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[CVT_X:[0-9]+]] = fir.convert %[[ARG]] : (!fir.logical<4>) -> i1
!CHECK:   %[[AND_XYZ:[0-9]+]] = arith.andi %[[CVT_X]], %[[AND_YZ]] : i1
!CHECK:   %[[RET:[0-9]+]] = fir.convert %[[AND_XYZ]] : (i1) -> !fir.logical<4>
!CHECK:   omp.yield(%[[RET]] : !fir.logical<4>)
!CHECK: }


subroutine f01(x, y, z)
  implicit none
  logical :: x, y, z

  !$omp atomic update
  x = x .or. y .or. z
end

!CHECK-LABEL: func.func @_QPf01
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[Z:[0-9]+]]:2 = hlfir.declare %arg2
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<!fir.logical<4>>
!CHECK: %[[LOAD_Z:[0-9]+]] = fir.load %[[Z]]#0 : !fir.ref<!fir.logical<4>>
!CHECK: %[[CVT_Y:[0-9]+]] = fir.convert %[[LOAD_Y]] : (!fir.logical<4>) -> i1
!CHECK: %[[CVT_Z:[0-9]+]] = fir.convert %[[LOAD_Z]] : (!fir.logical<4>) -> i1
!CHECK: %[[OR_YZ:[0-9]+]] = arith.ori %[[CVT_Y]], %[[CVT_Z]] : i1
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[CVT_X:[0-9]+]] = fir.convert %[[ARG]] : (!fir.logical<4>) -> i1
!CHECK:   %[[OR_XYZ:[0-9]+]] = arith.ori %[[CVT_X]], %[[OR_YZ]] : i1
!CHECK:   %[[RET:[0-9]+]] = fir.convert %[[OR_XYZ]] : (i1) -> !fir.logical<4>
!CHECK:   omp.yield(%[[RET]] : !fir.logical<4>)
!CHECK: }


subroutine f02(x, y, z)
  implicit none
  logical :: x, y, z

  !$omp atomic update
  x = x .eqv. y .eqv. z
end

!CHECK-LABEL: func.func @_QPf02
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[Z:[0-9]+]]:2 = hlfir.declare %arg2
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<!fir.logical<4>>
!CHECK: %[[LOAD_Z:[0-9]+]] = fir.load %[[Z]]#0 : !fir.ref<!fir.logical<4>>
!CHECK: %[[CVT_Y:[0-9]+]] = fir.convert %[[LOAD_Y]] : (!fir.logical<4>) -> i1
!CHECK: %[[CVT_Z:[0-9]+]] = fir.convert %[[LOAD_Z]] : (!fir.logical<4>) -> i1
!CHECK: %[[EQV_YZ:[0-9]+]] = arith.cmpi eq, %[[CVT_Y]], %[[CVT_Z]] : i1
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[CVT_X:[0-9]+]] = fir.convert %[[ARG]] : (!fir.logical<4>) -> i1
!CHECK:   %[[EQV_XYZ:[0-9]+]] = arith.cmpi eq, %[[CVT_X]], %[[EQV_YZ]] : i1
!CHECK:   %[[RET:[0-9]+]] = fir.convert %[[EQV_XYZ]] : (i1) -> !fir.logical<4>
!CHECK:   omp.yield(%[[RET]] : !fir.logical<4>)
!CHECK: }


subroutine f03(x, y, z)
  implicit none
  logical :: x, y, z

  !$omp atomic update
  x = x .neqv. y .neqv. z
end

!CHECK-LABEL: func.func @_QPf03
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[Z:[0-9]+]]:2 = hlfir.declare %arg2
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<!fir.logical<4>>
!CHECK: %[[LOAD_Z:[0-9]+]] = fir.load %[[Z]]#0 : !fir.ref<!fir.logical<4>>
!CHECK: %[[CVT_Y:[0-9]+]] = fir.convert %[[LOAD_Y]] : (!fir.logical<4>) -> i1
!CHECK: %[[CVT_Z:[0-9]+]] = fir.convert %[[LOAD_Z]] : (!fir.logical<4>) -> i1
!CHECK: %[[NEQV_YZ:[0-9]+]] = arith.cmpi ne, %[[CVT_Y]], %[[CVT_Z]] : i1
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[CVT_X:[0-9]+]] = fir.convert %[[ARG]] : (!fir.logical<4>) -> i1
!CHECK:   %[[NEQV_XYZ:[0-9]+]] = arith.cmpi ne, %[[CVT_X]], %[[NEQV_YZ]] : i1
!CHECK:   %[[RET:[0-9]+]] = fir.convert %[[NEQV_XYZ]] : (i1) -> !fir.logical<4>
!CHECK:   omp.yield(%[[RET]] : !fir.logical<4>)
!CHECK: }


subroutine f04(x, a, b, c)
  implicit none
  logical(kind=4) :: x
  logical(kind=8) :: a, b, c

  !$omp atomic update
  x = ((b .and. a) .and. x) .and. c
end

!CHECK-LABEL: func.func @_QPf04
!CHECK: %[[A:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[B:[0-9]+]]:2 = hlfir.declare %arg2
!CHECK: %[[C:[0-9]+]]:2 = hlfir.declare %arg3
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[LOAD_B:[0-9]+]] = fir.load %[[B]]#0 : !fir.ref<!fir.logical<8>>
!CHECK: %[[LOAD_A:[0-9]+]] = fir.load %[[A]]#0 : !fir.ref<!fir.logical<8>>
!CHECK: %[[CVT_B:[0-9]+]] = fir.convert %[[LOAD_B]] : (!fir.logical<8>) -> i1
!CHECK: %[[CVT_A:[0-9]+]] = fir.convert %[[LOAD_A]] : (!fir.logical<8>) -> i1
!CHECK: %[[AND_BA:[0-9]+]] = arith.andi %[[CVT_B]], %[[CVT_A]] : i1
!CHECK: %[[LOAD_C:[0-9]+]] = fir.load %[[C]]#0 : !fir.ref<!fir.logical<8>>
!CHECK: %[[CVT_C:[0-9]+]] = fir.convert %[[LOAD_C]] : (!fir.logical<8>) -> i1
!CHECK: %[[AND_BAC:[0-9]+]] = arith.andi %[[AND_BA]], %[[CVT_C]] : i1
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[CVT8_X:[0-9]+]] = fir.convert %[[ARG]] : (!fir.logical<4>) -> !fir.logical<8>
!CHECK:   %[[CVT_X:[0-9]+]] = fir.convert %[[CVT8_X]] : (!fir.logical<8>) -> i1
!CHECK:   %[[AND_XBAC:[0-9]+]] = arith.andi %[[CVT_X]], %[[AND_BAC]] : i1

!CHECK:   %[[RET:[0-9]+]] = fir.convert %[[AND_XBAC]] : (i1) -> !fir.logical<4>
!CHECK:   omp.yield(%[[RET]] : !fir.logical<4>)
!CHECK: }
