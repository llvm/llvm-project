! RUN: bbc -emit-fir %s -o - | FileCheck %s
 
! FRACTION
! CHECK-LABE: fraction_test
subroutine fraction_test

    real(kind=4) :: x1 = 178.1387e-4
    real(kind=8) :: x2 = 178.1387e-4
    real(kind=10) :: x3 = 178.1387e-4
    real(kind=16) :: x4 = 178.1387e-4
  ! CHECK: %[[r0:.*]] = fir.address_of(@_QFfraction_testEx1) : !fir.ref<f32>
  ! CHECK: %[[r1:.*]] = fir.address_of(@_QFfraction_testEx2) : !fir.ref<f64>
  ! CHECK: %[[r2:.*]] = fir.address_of(@_QFfraction_testEx3) : !fir.ref<f80>
  ! CHECK: %[[r3:.*]] = fir.address_of(@_QFfraction_testEx4) : !fir.ref<f128>
  
    x1 = fraction(x1)
  ! CHECK: %[[temp0:.*]] = fir.load %[[r0:.*]] : !fir.ref<f32>
  ! CHECK: %[[result0:.*]] = fir.call @_FortranAFraction4(%[[temp0:.*]]) : (f32) -> f32
  ! CHECK: fir.store %[[result0:.*]] to %[[r0:.*]] : !fir.ref<f32>
  
    x2 = fraction(x2)
  ! CHECK: %[[temp1:.*]] = fir.load %[[r1:.*]] : !fir.ref<f64>
  ! CHECK: %[[result1:.*]] = fir.call @_FortranAFraction8(%[[temp1:.*]]) : (f64) -> f64
  ! CHECK: fir.store %[[result1:.*]] to %[[r1:.*]] : !fir.ref<f64>
  
    x3 = fraction(x3)
  ! CHECK: %[[temp2:.*]] = fir.load %[[r2:.*]] : !fir.ref<f80>
  ! CHECK: %[[result2:.*]] = fir.call @_FortranAFraction10(%[[temp2:.*]]) : (f80) -> f80
  ! CHECK: fir.store %[[result2:.*]] to %[[r2:.*]] : !fir.ref<f80>
  
    x4 = fraction(x4)
  ! CHECK: %[[temp3:.*]] = fir.load %[[r3:.*]] : !fir.ref<f128>
  ! CHECK: %[[result3:.*]] = fir.call @_FortranAFraction16(%[[temp3:.*]]) : (f128) -> f128
  ! CHECK: fir.store %[[result3:.*]] to %[[r3:.*]] : !fir.ref<f128>
  end subroutine fraction_test
