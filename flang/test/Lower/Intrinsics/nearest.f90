! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: nearest_test1
subroutine nearest_test1(x, s)
    real :: x, s, res
  ! CHECK: %[[res:.*]] = fir.alloca f32 {bindc_name = "res", uniq_name = "_QFnearest_test1Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK: %[[s:.*]] = fir.load %arg1 : !fir.ref<f32>
  ! CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
  ! CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[s]], %[[zero]] : f32
  ! CHECK: %[[pos:.*]] = arith.select %[[cmp]], %true, %false : i1
    res = nearest(x, s)
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranANearest4(%[[x]], %[[pos]]) : (f32, i1) -> f32
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f32>
  end subroutine nearest_test1
  
  ! CHECK-LABEL: nearest_test2
  subroutine nearest_test2(x, s)
    real(kind=8) :: x, s, res
  ! CHECK: %[[res:.*]] = fir.alloca f64 {bindc_name = "res", uniq_name = "_QFnearest_test2Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f64>
  ! CHECK: %[[s:.*]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
  ! CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[s]], %[[zero]] : f64
  ! CHECK: %[[pos:.*]] = arith.select %[[cmp]], %true, %false : i1
    res = nearest(x, s)
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranANearest8(%[[x]], %[[pos]]) : (f64, i1) -> f64
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f64>
  end subroutine nearest_test2
  
  ! CHECK-LABEL: nearest_test3
  subroutine nearest_test3(x, s)
    real(kind=10) :: x, s, res
  ! CHECK: %[[res:.*]] = fir.alloca f80 {bindc_name = "res", uniq_name = "_QFnearest_test3Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f80>
  ! CHECK: %[[s:.*]] = fir.load %arg1 : !fir.ref<f80>
  ! CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f80
  ! CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[s]], %[[zero]] : f80
  ! CHECK: %[[pos:.*]] = arith.select %[[cmp]], %true, %false : i1
    res = nearest(x, s)
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranANearest10(%[[x]], %[[pos]]) : (f80, i1) -> f80
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f80>
  end subroutine nearest_test3
  
  ! CHECK-LABEL: nearest_test4
  subroutine nearest_test4(x, s)
    real(kind=16) :: x, s, res
  ! CHECK: %[[res:.*]] = fir.alloca f128 {bindc_name = "res", uniq_name = "_QFnearest_test4Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f128>
  ! CHECK: %[[s:.*]] = fir.load %arg1 : !fir.ref<f128>
  ! CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f128
  ! CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[s]], %[[zero]] : f128
  ! CHECK: %[[pos:.*]] = arith.select %[[cmp]], %true, %false : i1
    res = nearest(x, s)
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranANearest16(%[[x]], %[[pos]]) : (f128, i1) -> f128
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f128>
  end subroutine nearest_test4
  
  ! CHECK-LABEL: nearest_test5
  subroutine nearest_test5(x, s)
    real(kind=16) :: x, res
  ! CHECK: %[[res:.*]] = fir.alloca f128 {bindc_name = "res", uniq_name = "_QFnearest_test5Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f128>
    real :: s
  ! CHECK: %[[s:.*]] = fir.load %arg1 : !fir.ref<f32>
  ! CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
  ! CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[s]], %[[zero]] : f32
  ! CHECK: %[[pos:.*]] = arith.select %[[cmp]], %true, %false : i1
    res = nearest(x, s)
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranANearest16(%[[x]], %[[pos]]) : (f128, i1) -> f128
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f128>
  end subroutine nearest_test5
