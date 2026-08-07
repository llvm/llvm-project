!RUN: %flang_fc1 -emit-hlfir -ffast-math -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

subroutine f00(x, y)
  implicit none
  real :: x, y

  !$omp atomic update
  x = ((x + 1) + y) + 2
end

!CHECK-LABEL: func.func @_QPf00
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %cst = arith.constant 1.000000e+00 : f32
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<f32>
!CHECK: %[[Y_1:[0-9]+]] = arith.addf %cst, %[[LOAD_Y]] fastmath<fast> : f32
!CHECK: %cst_0 = arith.constant 2.000000e+00 : f32
!CHECK: %[[Y_1_2:[0-9]+]] = arith.addf %[[Y_1]], %cst_0 fastmath<fast> : f32
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<f32> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: f32):
!CHECK:   %[[ARG_P:[0-9]+]] = arith.addf %[[ARG]], %[[Y_1_2]] fastmath<fast> : f32
!CHECK:   omp.yield(%[[ARG_P]] : f32)
!CHECK: }


subroutine f01(x, y, z)
  implicit none
  complex :: x, y, z

  !$omp atomic update
  x = (x + y) + z
end

!CHECK-LABEL: func.func @_QPf01
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[Z:[0-9]+]]:2 = hlfir.declare %arg2
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<complex<f32>>
!CHECK: %[[LOAD_Z:[0-9]+]] = fir.load %[[Z]]#0 : !fir.ref<complex<f32>>
!CHECK: %[[Y_Z:[0-9]+]] = fir.addc %[[LOAD_Y]], %[[LOAD_Z]] {fastmath = #arith.fastmath<fast>} : complex<f32>
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<complex<f32>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: complex<f32>):
!CHECK:   %[[ARG_P:[0-9]+]] = fir.addc %[[ARG]], %[[Y_Z]] {fastmath = #arith.fastmath<fast>} : complex<f32>
!CHECK:   omp.yield(%[[ARG_P]] : complex<f32>)
!CHECK: }


subroutine f02(x, y)
  implicit none
  complex :: x
  real :: y

  !$omp atomic update
  x = (real(x) + y) + 1
end

!CHECK-LABEL: func.func @_QPf02
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[Y:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[LOAD_Y:[0-9]+]] = fir.load %[[Y]]#0 : !fir.ref<f32>
!CHECK: %cst = arith.constant 1.000000e+00 : f32
!CHECK: %[[Y_1:[0-9]+]] = arith.addf %[[LOAD_Y]], %cst fastmath<fast> : f32
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<complex<f32>> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: complex<f32>):
!CHECK: %[[ARG_X:[0-9]+]] = fir.extract_value %[[ARG]], [0 : index] : (complex<f32>) -> f32
!CHECK: %[[ARG_P:[0-9]+]] = arith.addf %[[ARG_X]], %[[Y_1]] fastmath<fast> : f32
!CHECK: %cst_0 = arith.constant 0.000000e+00 : f32
!CHECK: %[[CPLX:[0-9]+]] = fir.undefined complex<f32>
!CHECK: %[[CPLX_I:[0-9]+]] = fir.insert_value %[[CPLX]], %[[ARG_P]], [0 : index] : (complex<f32>, f32) -> complex<f32>
!CHECK: %[[CPLX_R:[0-9]+]] = fir.insert_value %[[CPLX_I]], %cst_0, [1 : index] : (complex<f32>, f32) -> complex<f32>
!CHECK:   omp.yield(%[[CPLX_R]] : complex<f32>)
!CHECK: }


subroutine f03(x, a, b, c)
  implicit none
  real(kind=4) :: x
  real(kind=8) :: a, b, c

  !$omp atomic update
  x = ((b + a) + x) + c
end

!CHECK-LABEL: func.func @_QPf03
!CHECK: %[[A:[0-9]+]]:2 = hlfir.declare %arg1
!CHECK: %[[B:[0-9]+]]:2 = hlfir.declare %arg2
!CHECK: %[[C:[0-9]+]]:2 = hlfir.declare %arg3
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare %arg0
!CHECK: %[[LOAD_B:[0-9]+]] = fir.load %[[B]]#0 : !fir.ref<f64>
!CHECK: %[[LOAD_A:[0-9]+]] = fir.load %[[A]]#0 : !fir.ref<f64>
!CHECK: %[[A_B:[0-9]+]] = arith.addf %[[LOAD_B]], %[[LOAD_A]] fastmath<fast> : f64
!CHECK: %[[LOAD_C:[0-9]+]] = fir.load %[[C]]#0 : !fir.ref<f64>
!CHECK: %[[A_B_C:[0-9]+]] = arith.addf %[[A_B]], %[[LOAD_C]] fastmath<fast> : f64
!CHECK: omp.atomic.update memory_order(relaxed) %[[X]]#0 : !fir.ref<f32> {
!CHECK: ^bb0(%[[ARG:arg[0-9]+]]: f32):
!CHECK:   %[[ARG_8:[0-9]+]] = fir.convert %[[ARG]] : (f32) -> f64
!CHECK:   %[[ARG_P:[0-9]+]] = arith.addf %[[ARG_8]], %[[A_B_C]] fastmath<fast> : f64
!CHECK:   %[[ARG_4:[0-9]+]] = fir.convert %[[ARG_P]] : (f64) -> f32
!CHECK:   omp.yield(%[[ARG_4]] : f32)
!CHECK: }
