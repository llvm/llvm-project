! Tests for -finit-local= with REAL(16) and COMPLEX(16) (IEEE f128).
! These types require f128 math support, which is not available on AIX.
!
! REQUIRES: flang-supports-f128-math
!
! RUN: bbc -emit-hlfir -finit-local=zero  -o - %s | FileCheck --check-prefix=ZERO  %s
! RUN: bbc -emit-hlfir -finit-local=nan   -o - %s | FileCheck --check-prefix=NAN   %s
! RUN: bbc -emit-hlfir -finit-local=snan  -o - %s | FileCheck --check-prefix=SNAN  %s
! RUN: bbc -emit-hlfir -finit-local=0xAA  -o - %s | FileCheck --check-prefix=HEX   %s

! ---------------------------------------------------------------------------
! REAL(16) -- 16-byte FP (f128); hex uses 128-bit APInt splat + bitcast
! 0xAA * 16 bytes = -113427455640312821154458202477256070486 (signed i128)
! ---------------------------------------------------------------------------
subroutine test_real16(res)
  real(16) :: res
  real(16) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_real16
! ZERO: fir.zero_bits f128
! ZERO: fir.store {{.*}} : !fir.ref<f128>

! NAN-LABEL:  func.func @_QPtest_real16
! NAN:  arith.constant {{.*}} : f128
! NAN:  fir.store {{.*}} : !fir.ref<f128>

! SNAN-LABEL: func.func @_QPtest_real16
! SNAN: arith.constant {{.*}} : f128
! SNAN: fir.store {{.*}} : !fir.ref<f128>

! HEX-LABEL:  func.func @_QPtest_real16
! HEX:  arith.constant -113427455640312821154458202477256070486 : i128
! HEX:  arith.bitcast {{.*}} : i128 to f128
! HEX:  fir.store {{.*}} : !fir.ref<f128>

! ---------------------------------------------------------------------------
! COMPLEX(16) -- two f128 parts; hex uses 128-bit APInt splat + bitcast
! 0xAA * 16 bytes = -113427455640312821154458202477256070486 (signed i128)
! ---------------------------------------------------------------------------
subroutine test_complex16(res)
  complex(16) :: res
  complex(16) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_complex16
! ZERO: fir.zero_bits !fir.complex<16>
! ZERO: fir.store {{.*}} : !fir.ref<!fir.complex<16>>

! NAN-LABEL:  func.func @_QPtest_complex16
! NAN:  arith.constant {{.*}} : f128
! NAN:  complex.create {{.*}}, {{.*}} : f128
! NAN:  fir.store {{.*}} : !fir.ref<!fir.complex<16>>

! SNAN-LABEL: func.func @_QPtest_complex16
! SNAN: arith.constant {{.*}} : f128
! SNAN: complex.create {{.*}}, {{.*}} : f128
! SNAN: fir.store {{.*}} : !fir.ref<!fir.complex<16>>

! HEX-LABEL:  func.func @_QPtest_complex16
! HEX:  arith.constant -113427455640312821154458202477256070486 : i128
! HEX:  arith.bitcast {{.*}} : i128 to f128
! HEX:  complex.create {{.*}}, {{.*}} : f128
! HEX:  fir.store {{.*}} : !fir.ref<!fir.complex<16>>
