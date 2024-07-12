! Check that correct runtime calls are used.
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck --check-prefixes=CHECK-RUNTIME %s
! RUN: %flang_fc1 -mllvm -math-runtime=precise -emit-fir %s -o - | FileCheck --check-prefixes=CHECK-RUNTIME %s

! Check that the correct math dialect operations are used.
! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK-NORMAL %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK-NORMAL %s

function test_real4(x)
  real :: x, test_real4
  test_real4 = atan(x)
end function

! CHECK-RUNTIME: {{%[A-Za-z0-9._]+}} = fir.call @atanf({{.*}}) {{.*}}: (f32) -> f32
! CHECK-NORMAL: {{%[A-Za-z0-9._]+}} = math.atan {{.*}} {{.*}}: f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = atan(x)
end function

! CHECK-RUNTIME: {{%[A-Za-z0-9._]+}} = fir.call @atan({{.*}}) {{.*}}: (f64) -> f64
! CHECK-NORMAL: {{%[A-Za-z0-9._]+}} = math.atan {{.*}} {{.*}}: f64

function test_complex4(x)
  complex :: x, test_complex4
  test_complex4 = atan(x)
end function

! CHECK-RUNTIME: {{%[A-Za-z0-9._]+}} = fir.call @catanf({{.*}}) {{.*}}: (!fir.complex<4>) -> !fir.complex<4>
! CHECK-NORMAL: {{%[A-Za-z0-9._]+}} = fir.call @catanf({{.*}}) {{.*}}: (!fir.complex<4>) -> !fir.complex<4>

function test_complex8(x)
  complex(kind=8) :: x, test_complex8
  test_complex8 = atan(x)
end function

! CHECK-RUNTIME: {{%[A-Za-z0-9._]+}} = fir.call @catan({{.*}}) {{.*}}: (!fir.complex<8>) -> !fir.complex<8>
! CHECK-NORMAL: {{%[A-Za-z0-9._]+}} = fir.call @catan({{.*}}) {{.*}}: (!fir.complex<8>) -> !fir.complex<8>

function test_real4_2(y, x)
  real :: y, x, test_real4_2
  test_real4_2 = atan(y, x)
end function

! CHECK-RUNTIME: {{%[A-Za-z0-9._]+}} = fir.call @atan2f({{.*}}) {{.*}}: (f32, f32) -> f32
! CHECK-NORMAL: {{%[A-Za-z0-9._]+}} = math.atan2 {{.*}} {{.*}}: f32

function test_real8_2(y, x)
  real(8) :: y, x, test_real8_2
  test_real8_2 = atan(y, x)
end function

! CHECK-RUNTIME: {{%[A-Za-z0-9._]+}} = fir.call @atan2({{.*}}) {{.*}}: (f64, f64) -> f64
! CHECK-NORMAL: {{%[A-Za-z0-9._]+}} = math.atan2 {{.*}} {{.*}}: f64
