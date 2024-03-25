! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL %s

function test_real4(x)
  real :: x, test_real4
  test_real4 = nint(x, 4) + nint(x, 8)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.lround.i32.f32({{%[A-Za-z0-9._]+}}) {{.*}}: (f32) -> i32
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.lround.i64.f32({{%[A-Za-z0-9._]+}}) {{.*}}: (f32) -> i64

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = nint(x, 4) + nint(x, 8)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.lround.i32.f64({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> i32
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.lround.i64.f64({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> i64

! ALL-DAG: func.func private @llvm.lround.i32.f32(f32) -> i32 attributes {fir.bindc_name = "llvm.lround.i32.f32", fir.runtime}
! ALL-DAG: func.func private @llvm.lround.i64.f32(f32) -> i64 attributes {fir.bindc_name = "llvm.lround.i64.f32", fir.runtime}
! ALL-DAG: func.func private @llvm.lround.i32.f64(f64) -> i32 attributes {fir.bindc_name = "llvm.lround.i32.f64", fir.runtime}
! ALL-DAG: func.func private @llvm.lround.i64.f64(f64) -> i64 attributes {fir.bindc_name = "llvm.lround.i64.f64", fir.runtime}
