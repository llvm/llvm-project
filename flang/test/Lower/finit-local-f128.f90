! Tests for -finit-local= with REAL(16) and COMPLEX(16) (IEEE f128).
! These types require f128 math support, which is not available on AIX.
!
! REQUIRES: flang-supports-f128-math
!
! RUN: %flang_fc1 -emit-hlfir -finit-local=zero  %s -o - | FileCheck --check-prefix=ZERO  %s
! RUN: %flang_fc1 -emit-hlfir -finit-local=0xAA  %s -o - | FileCheck --check-prefix=HEX   %s

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
! ZERO: fir.zero_bits complex<f128>
! ZERO: fir.store {{.*}} : !fir.ref<complex<f128>>



! HEX-LABEL:  func.func @_QPtest_complex16
! HEX:  arith.constant -113427455640312821154458202477256070486 : i128
! HEX:  arith.bitcast {{.*}} : i128 to f128
! HEX:  complex.create {{.*}}, {{.*}} : complex<f128>
! HEX:  fir.store {{.*}} : !fir.ref<complex<f128>>
