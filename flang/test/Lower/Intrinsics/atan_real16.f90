! REQUIRES: flang-supports-f128-math
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

function test_real16(x)
  real(16) :: x, test_real16
  test_real16 = atan(x)
end function

! CHECK-LABEL: @_QPtest_real16
! CHECK: fir.call @_FortranAAtanF128({{.*}}){{.*}}: (f128) -> f128

function test_real16_2(y, x)
  real(16) :: y, x, test_real16_2
  test_real16_2 = atan(y, x)
end function

! CHECK-LABEL: @_QPtest_real16_2
! CHECK: fir.call @_FortranAAtan2F128({{.*}}){{.*}}: (f128, f128) -> f128
