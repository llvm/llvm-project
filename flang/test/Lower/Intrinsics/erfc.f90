! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL,FAST %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL,PRECISE %s

function test_real4(x)
  real :: x, test_real4
  test_real4 = erfc(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.erfc {{%[A-Za-z0-9._]+}} {{.*}}: f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.erfc {{%[A-Za-z0-9._]+}} {{.*}}: f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @erfcf({{%[A-Za-z0-9._]+}}) {{.*}}: (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = erfc(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.erfc {{%[A-Za-z0-9._]+}} {{.*}}: f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.erfc {{%[A-Za-z0-9._]+}} {{.*}}: f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @erfc({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> f64

function dtest_real8(x)
  real(8) :: x, dtest_real8
  dtest_real8 = derfc(x)
end function

! ALL-LABEL: @_QPdtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.erfc {{%[A-Za-z0-9._]+}} {{.*}}: f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.erfc {{%[A-Za-z0-9._]+}} {{.*}}: f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @erfc({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> f64
