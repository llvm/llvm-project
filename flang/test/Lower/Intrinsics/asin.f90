! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL %s

function test_real4(x)
  real :: x, test_real4
  test_real4 = asin(x)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @asinf({{%[A-Za-z0-9._]+}}) {{.*}}: (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = asin(x)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @asin({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> f64

function test_complex4(x)
  complex :: x, test_complex4
  test_complex4 = asin(x)
end function

! ALL-LABEL: @_QPtest_complex4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @casinf({{%[A-Za-z0-9._]+}}) {{.*}}: (!fir.complex<4>) -> !fir.complex<4>

function test_complex8(x)
  complex(kind=8) :: x, test_complex8
  test_complex8 = asin(x)
end function

! ALL-LABEL: @_QPtest_complex8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @casin({{%[A-Za-z0-9._]+}}) {{.*}}: (!fir.complex<8>) -> !fir.complex<8>

