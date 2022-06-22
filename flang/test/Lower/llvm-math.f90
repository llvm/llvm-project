! RUN: bbc -emit-fir %s -o - --math-runtime=llvm --outline-intrinsics | FileCheck %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=llvm -mllvm -outline-intrinsics %s -o - | FileCheck %s

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

      SUBROUTINE ATAN_WRAPPER(IN, OUT)
      DOUBLE PRECISION IN
      OUT = DATAN(IN)
      RETURN
      END

! CHECK:       func private @fir.atan.f64.f64(%arg0: f64)
! CHECK-NEXT:   %0 = fir.call @atan(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }

      SUBROUTINE EXP_WRAPPER(IN, OUT)
      DOUBLE PRECISION IN
      OUT = DEXP(IN)
      RETURN
      END

! CHECK:       func private @fir.exp.f64.f64(%arg0: f64)
! CHECK-NEXT:   %0 = fir.call @llvm.exp.f64(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }

      SUBROUTINE SINH_WRAPPER(IN, OUT)
      DOUBLE PRECISION IN
      OUT = DSINH(IN)
      RETURN
      END

! CHECK:       func private @fir.sinh.f64.f64(%arg0: f64)
! CHECK-NEXT:   %0 = fir.call @sinh(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }

      SUBROUTINE COSH_WRAPPER(IN, OUT)
      DOUBLE PRECISION IN
      OUT = DCOSH(IN)
      RETURN
      END

! CHECK:       func private @fir.cosh.f64.f64(%arg0: f64)
! CHECK-NEXT:   %0 = fir.call @cosh(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }


      SUBROUTINE ATANF_WRAPPER(IN, OUT)
      REAL IN
      OUT = ATAN(IN)
      RETURN
      END

! CHECK:       func private @fir.atan.f32.f32(%arg0: f32)
! CHECK-NEXT:   %0 = fir.call @atanf(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
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

      SUBROUTINE SINHF_WRAPPER(IN, OUT)
      REAL IN
      OUT = SINH(IN)
      RETURN
      END

! CHECK:       func private @fir.sinh.f32.f32(%arg0: f32)
! CHECK-NEXT:   %0 = fir.call @sinhf(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
! CHECK-NEXT: }

      SUBROUTINE COSHF_WRAPPER(IN, OUT)
      REAL IN
      OUT = COSH(IN)
      RETURN
      END

! CHECK:       func private @fir.cosh.f32.f32(%arg0: f32)
! CHECK-NEXT:   %0 = fir.call @coshf(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
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

      SUBROUTINE SQRT_WRAPPER(IN, OUT)
      REAL :: IN, OUT
      OUT = SQRT(IN)
      END SUBROUTINE

! CHECK-LABEL: func private @fir.sqrt.f32.f32(%arg0: f32)
! CHECK-NEXT:  %0 = fir.call @llvm.sqrt.f32(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
! CHECK-NEXT: }

      SUBROUTINE SQRTD_WRAPPER(IN, OUT)
      REAL(KIND=8) :: IN, OUT
      OUT = SQRT(IN)
      END SUBROUTINE

! CHECK-LABEL: func private @fir.sqrt.f64.f64(%arg0: f64)
! CHECK-NEXT:  %0 = fir.call @llvm.sqrt.f64(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }

      SUBROUTINE COS_WRAPPER(IN, OUT)
      REAL :: IN, OUT
      OUT = COS(IN)
      END SUBROUTINE

! CHECK-LABEL: func private @fir.cos.f32.f32(%arg0: f32)
! CHECK-NEXT:  %0 = fir.call @llvm.cos.f32(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
! CHECK-NEXT: }

      SUBROUTINE COSD_WRAPPER(IN, OUT)
      REAL(KIND=8) :: IN, OUT
      OUT = COS(IN)
      END SUBROUTINE

! CHECK-LABEL: func private @fir.cos.f64.f64(%arg0: f64)
! CHECK-NEXT:  %0 = fir.call @llvm.cos.f64(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }

      SUBROUTINE SIN_WRAPPER(IN, OUT)
      REAL :: IN, OUT
      OUT = SIN(IN)
      END SUBROUTINE

! CHECK-LABEL: func private @fir.sin.f32.f32(%arg0: f32)
! CHECK-NEXT:  %0 = fir.call @llvm.sin.f32(%arg0) : (f32) -> f32
! CHECK-NEXT:   return %0 : f32
! CHECK-NEXT: }

      SUBROUTINE SIND_WRAPPER(IN, OUT)
      REAL(KIND=8) :: IN, OUT
      OUT = SIN(IN)
      END SUBROUTINE

! CHECK-LABEL: func private @fir.sin.f64.f64(%arg0: f64)
! CHECK-NEXT:  %0 = fir.call @llvm.sin.f64(%arg0) : (f64) -> f64
! CHECK-NEXT:   return %0 : f64
! CHECK-NEXT: }
