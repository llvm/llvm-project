! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL,FAST %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL,PRECISE %s

function test_real4(x, y)
  real :: x, y, test_real4
  test_real4 = sign(x, y)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @copysignf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f32, f32) -> f32

function test_real8(x, y)
  real(8) :: x, y, test_real8
  test_real8 = sign(x, y)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f64, f64) -> f64

function test_real10(x, y)
  real(10) :: x, y, test_real10
  test_real10 = sign(x, y)
end function

! ALL-LABEL: @_QPtest_real10
! FAST: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f80
! RELAXED: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f80
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @copysignl({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f80, f80) -> f80

function test_real16(x, y)
  real(16) :: x, y, test_real16
  test_real16 = sign(x, y)
end function

! ALL-LABEL: @_QPtest_real16
! FAST: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f128
! RELAXED: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f128
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.copysign.f128({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f128, f128) -> f128

! PRECISE-DAG: func.func private @copysignf(f32, f32) -> f32 attributes {fir.bindc_name = "copysignf", fir.runtime}
! PRECISE-DAG: func.func private @copysign(f64, f64) -> f64 attributes {fir.bindc_name = "copysign", fir.runtime}
! PRECISE-DAG: func.func private @copysignl(f80, f80) -> f80 attributes {fir.bindc_name = "copysignl", fir.runtime}
! PRECISE-DAG: func.func private @llvm.copysign.f128(f128, f128) -> f128 attributes {fir.bindc_name = "llvm.copysign.f128", fir.runtime}
