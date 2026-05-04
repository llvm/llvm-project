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
!CHECK: %[[AND_YZ:[0-9]+]] = fir.logical_and %[[LOAD_Y]], %[[LOAD_Z]] : !fir.logical<4>
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[RET:[0-9]+]] = fir.logical_and %[[ARG]], %[[AND_YZ]] : !fir.logical<4>
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
!CHECK: %[[OR_YZ:[0-9]+]] = fir.logical_or %[[LOAD_Y]], %[[LOAD_Z]] : !fir.logical<4>
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[RET:[0-9]+]] = fir.logical_or %[[ARG]], %[[OR_YZ]] : !fir.logical<4>
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
!CHECK: %[[EQV_YZ:[0-9]+]] = fir.eqv %[[LOAD_Y]], %[[LOAD_Z]] : !fir.logical<4>
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[RET:[0-9]+]] = fir.eqv %[[ARG]], %[[EQV_YZ]] : !fir.logical<4>
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
!CHECK: %[[NEQV_YZ:[0-9]+]] = fir.neqv %[[LOAD_Y]], %[[LOAD_Z]] : !fir.logical<4>
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[RET:[0-9]+]] = fir.neqv %[[ARG]], %[[NEQV_YZ]] : !fir.logical<4>
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
!CHECK: %[[AND_BA:[0-9]+]] = fir.logical_and %[[LOAD_B]], %[[LOAD_A]] : !fir.logical<8>
!CHECK: %[[LOAD_C:[0-9]+]] = fir.load %[[C]]#0 : !fir.ref<!fir.logical<8>>
!CHECK: %[[AND_BAC:[0-9]+]] = fir.logical_and %[[AND_BA]], %[[LOAD_C]] : !fir.logical<8>
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<!fir.logical<4>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: !fir.logical<4>):
!CHECK:   %[[CVT8_X:[0-9]+]] = fir.convert %[[ARG]] : (!fir.logical<4>) -> !fir.logical<8>
!CHECK:   %[[AND_XBAC:[0-9]+]] = fir.logical_and %[[CVT8_X]], %[[AND_BAC]] : !fir.logical<8>
!CHECK:   %[[RET:[0-9]+]] = fir.convert %[[AND_XBAC]] : (!fir.logical<8>) -> !fir.logical<4>
!CHECK:   omp.yield(%[[RET]] : !fir.logical<4>)
!CHECK: }
