! RUN: %flang_fc1 -emit-hlfir -o - -mllvm -math-runtime=fast %s \
! RUN: | FileCheck %s --check-prefixes=ALL,FAST%if target=x86_64{{.*}} %{,ALL-KIND10,FAST-KIND10%}%if flang-supports-f128-math %{,ALL-KIND16,FAST-KIND16%}

! RUN: %flang_fc1 -emit-hlfir -o - -mllvm -math-runtime=relaxed %s \
! RUN: | FileCheck %s --check-prefixes=ALL,RELAXED%if target=x86_64{{.*}} %{,ALL-KIND10,RELAXED-KIND10%}%if flang-supports-f128-math %{,ALL-KIND16,RELAXED-KIND16%}

! RUN: %flang_fc1 -emit-hlfir -o - -mllvm -math-runtime=precise %s \
! RUN: | FileCheck %s --check-prefixes=ALL,PRECISE%if target=x86_64{{.*}} %{,ALL-KIND10,PRECISE-KIND10%}%if flang-supports-f128-math %{,ALL-KIND16,PRECISE-KIND16%}

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
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(kind10) :: x, y, test_real10
  test_real10 = sign(x, y)
end function

! ALL-KIND10-LABEL: @_QPtest_real10
! FAST-KIND10: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f80
! RELAXED-KIND10: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f80
! PRECISE-KIND10: {{%[A-Za-z0-9._]+}} = fir.call @copysignl({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f80, f80) -> f80

function test_real16(x, y)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind16) :: x, y, test_real16
  test_real16 = sign(x, y)
end function

! ALL-KIND16-LABEL: @_QPtest_real16
! FAST-KIND16: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f128
! RELAXED-KIND16: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f128
! PRECISE-KIND16: {{%[A-Za-z0-9._]+}} = fir.call @llvm.copysign.f128({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f128, f128) -> f128

! PRECISE-DAG: func.func private @copysignf(f32, f32) -> f32 attributes {fir.bindc_name = "copysignf", fir.runtime}
! PRECISE-DAG: func.func private @copysign(f64, f64) -> f64 attributes {fir.bindc_name = "copysign", fir.runtime}
! PRECISE-KIND10-DAG: func.func private @copysignl(f80, f80) -> f80 attributes {fir.bindc_name = "copysignl", fir.runtime}
! PRECISE-KIND16-DAG: func.func private @llvm.copysign.f128(f128, f128) -> f128 attributes {fir.bindc_name = "llvm.copysign.f128", fir.runtime}
