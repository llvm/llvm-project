! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL,FAST %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL,PRECISE %s

function test_real4(x, y, s, i, k)
  real :: x, y, test_real4
  integer(2) :: s
  integer(4) :: i
  integer(8) :: k
  test_real4 = x ** s + x ** y + x ** i + x ** k
end function

! ALL-LABEL: @_QPtest_real4
! ALL: [[STOI:%[A-Za-z0-9._]+]] = fir.convert {{%[A-Za-z0-9._]+}} : (i16) -> i32
! FAST: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32, i32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32, i32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @_FortranAFPow4i({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f32, i32) -> f32
! FAST: {{%[A-Za-z0-9._]+}} = math.powf {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.powf {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @powf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f32, f32) -> f32
! FAST: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32, i32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32, i32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @_FortranAFPow4i({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f32, i32) -> f32
! FAST: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32, i64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f32, i64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @_FortranAFPow4k({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f32, i64) -> f32

function test_real8(x, y, s, i, k)
  real(8) :: x, y, test_real8
  integer(2) :: s
  integer(4) :: i
  integer(8) :: k
  test_real8 = x ** s + x ** y + x ** i + x ** k
end function

! ALL-LABEL: @_QPtest_real8
! ALL: [[STOI:%[A-Za-z0-9._]+]] = fir.convert {{%[A-Za-z0-9._]+}} : (i16) -> i32
! FAST: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64, i32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64, i32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @_FortranAFPow8i({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f64, i32) -> f64
! FAST: {{%[A-Za-z0-9._]+}} = math.powf {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.powf {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @pow({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f64, f64) -> f64
! FAST: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64, i32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64, i32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @_FortranAFPow8i({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f64, i32) -> f64
! FAST: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64, i64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.fpowi {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} {{.*}}: f64, i64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @_FortranAFPow8k({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (f64, i64) -> f64

! PRECISE-DAG: func.func private @_FortranAFPow4i(f32, i32) -> f32 attributes {fir.bindc_name = "_FortranAFPow4i", fir.runtime}
! PRECISE-DAG: func.func private @powf(f32, f32) -> f32 attributes {fir.bindc_name = "powf", fir.runtime}
! PRECISE-DAG: func.func private @_FortranAFPow4k(f32, i64) -> f32 attributes {fir.bindc_name = "_FortranAFPow4k", fir.runtime}
! PRECISE-DAG: func.func private @_FortranAFPow8i(f64, i32) -> f64 attributes {fir.bindc_name = "_FortranAFPow8i", fir.runtime}
! PRECISE-DAG: func.func private @pow(f64, f64) -> f64 attributes {fir.bindc_name = "pow", fir.runtime}
! PRECISE-DAG: func.func private @_FortranAFPow8k(f64, i64) -> f64 attributes {fir.bindc_name = "_FortranAFPow8k", fir.runtime}
