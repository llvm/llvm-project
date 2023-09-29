! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL,FAST %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL,PRECISE %s

function test_real4(x)
  real :: x, test_real4
  test_real4 = abs(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.absf {{%[A-Za-z0-9._]+}} {{.*}}: f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.absf {{%[A-Za-z0-9._]+}} {{.*}}: f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @fabsf({{%[A-Za-z0-9._]+}}) {{.*}}: (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = abs(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.absf {{%[A-Za-z0-9._]+}} {{.*}}: f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.absf {{%[A-Za-z0-9._]+}} {{.*}}: f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @fabs({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> f64

function test_real16(x)
  real(16) :: x, test_real16
  test_real16 = abs(x)
end function
! ALL-LABEL: @_QPtest_real16
! FAST: {{%[A-Za-z0-9._]+}} = math.absf {{%[A-Za-z0-9._]+}} {{.*}}: f128
! RELAXED: {{%[A-Za-z0-9._]+}} = math.absf {{%[A-Za-z0-9._]+}} {{.*}}: f128
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.fabs.f128({{%[A-Za-z0-9._]+}}) {{.*}}: (f128) -> f128

function test_complex4(c)
  complex(4) :: c, test_complex4
  test_complex4 = abs(c)
end function
! ALL-LABEL: @_QPtest_complex4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @cabsf({{%[A-Za-z0-9._]+}}) {{.*}}: (!fir.complex<4>) -> f32

function test_complex8(c)
  complex(8) :: c, test_complex8
  test_complex8 = abs(c)
end function
! ALL-LABEL: @_QPtest_complex8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @cabs({{%[A-Za-z0-9._]+}}) {{.*}}: (!fir.complex<8>) -> f64

! PRECISE-DAG: func.func private @fabsf(f32) -> f32 attributes {fir.bindc_name = "fabsf", fir.runtime}
! PRECISE-DAG: func.func private @fabs(f64) -> f64 attributes {fir.bindc_name = "fabs", fir.runtime}
! PRECISE-DAG: func.func private @llvm.fabs.f128(f128) -> f128 attributes {fir.bindc_name = "llvm.fabs.f128", fir.runtime}
! PRECISE-DAG: func.func private @cabsf(!fir.complex<4>) -> f32 attributes {fir.bindc_name = "cabsf", fir.runtime}
! PRECISE-DAG: func.func private @cabs(!fir.complex<8>) -> f64 attributes {fir.bindc_name = "cabs", fir.runtime}
