! RUN: bbc -emit-fir %s -o - --math-runtime=llvm | FileCheck %s

      SUBROUTINE POW_WRAPPER(IN, IN2, OUT)
      DOUBLE PRECISION IN, IN2
      OUT = IN ** IN2
      RETURN
      END
      
! CHECK-LABEL: func @_QPpow_wrapper(
! CHECK-SAME: %{{.*}}: !fir.ref<f64>{{.*}}, %{{.*}}: !fir.ref<f64>{{.*}}, %{{.*}}: !fir.ref<f32>{{.*}}) {
! CHECK-NEXT:   %0 = fir.load %arg0 : !fir.ref<f64>
! CHECK-NEXT:   %1 = fir.load %arg1 : !fir.ref<f64>
! CHECK-NEXT:   %2 = fir.call @llvm.pow.f64(%0, %1) : (f64, f64) -> f64

      SUBROUTINE POWF_WRAPPER(IN, IN2, OUT)
      REAL IN, IN2
      OUT = IN ** IN2
      RETURN
      END

! CHECK-LABEL: func @_QPpowf_wrapper(
! CHECK-SAME: %{{.*}}: !fir.ref<f32>{{.*}}, %{{.*}}: !fir.ref<f32>{{.*}}, %{{.*}}: !fir.ref<f32>{{.*}}) {
! CHECK-NEXT:   %0 = fir.load %arg0 : !fir.ref<f32>
! CHECK-NEXT:   %1 = fir.load %arg1 : !fir.ref<f32>
! CHECK-NEXT:   %2 = fir.call @llvm.pow.f32(%0, %1) : (f32, f32) -> f32
