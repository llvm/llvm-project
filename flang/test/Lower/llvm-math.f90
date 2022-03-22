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

      SUBROUTINE EXP_WRAPPER(IN, OUT)
      DOUBLE PRECISION IN
      OUT = DEXP(IN)
      RETURN
      END

! CHECK:       func private @fir.exp.f64.f64(%arg0: f64)
! CHECK-NEXT:   %0 = fir.call @llvm.exp.f64(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }

      SUBROUTINE LOG_WRAPPER(IN, OUT)
      DOUBLE PRECISION IN, OUT
      OUT = DLOG(IN)
      RETURN
      END

! CHECK:       func private @fir.log.f64.f64(%arg0: f64)
! CHECK-NEXT:   %0 = fir.call @llvm.log.f64(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }

      SUBROUTINE LOG10_WRAPPER(IN, OUT)
      DOUBLE PRECISION IN, OUT
      OUT = DLOG10(IN)
      RETURN
      END

! CHECK:       func private @fir.log10.f64.f64(%arg0: f64)
! CHECK-NEXT:   %0 = fir.call @llvm.log10.f64(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }

      SUBROUTINE EXPF_WRAPPER(IN, OUT)
      REAL IN
      OUT = EXP(IN)
      RETURN
      END

! CHECK:       func private @fir.exp.f32.f32(%arg0: f32)
! CHECK-NEXT:   %0 = fir.call @llvm.exp.f32(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
! CHECK-NEXT: }

      SUBROUTINE LOGF_WRAPPER(IN, OUT)
      REAL IN, OUT
      OUT = LOG(IN)
      RETURN
      END

! CHECK:       func private @fir.log.f32.f32(%arg0: f32)
! CHECK-NEXT:   %0 = fir.call @llvm.log.f32(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
! CHECK-NEXT: }

      SUBROUTINE LOG10F_WRAPPER(IN, OUT)
      REAL IN, OUT
      OUT = LOG10(IN)
      RETURN
      END

! CHECK:       func private @fir.log10.f32.f32(%arg0: f32)
! CHECK-NEXT:   %0 = fir.call @llvm.log10.f32(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
! CHECK-NEXT: }
