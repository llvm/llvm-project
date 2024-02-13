! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL,FAST %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL,RELAXED %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL,PRECISE %s

function test_real4(x)
  real :: x, test_real4
  test_real4 = anint(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f32) -> f32
! RELAXED: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f32) -> f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.round.f32({{%[A-Za-z0-9._]+}}) {{.*}}: (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = anint(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f64) -> f64
! RELAXED: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f64) -> f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.round.f64({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> f64

function test_real10(x)
  real(10) :: x, test_real10
  test_real10 = anint(x)
end function

! ALL-LABEL: @_QPtest_real10
! FAST: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f80) -> f80
! RELAXED: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f80) -> f80
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.round.f80({{%[A-Za-z0-9._]+}}) {{.*}}: (f80) -> f80

! TODO: wait until fp128 is supported well in llvm.round
!function test_real16(x)
!  real(16) :: x, test_real16
!  test_real16 = anint(x)
!end function

! PRECISE-DAG: func.func private @llvm.round.f32(f32) -> f32 attributes {fir.bindc_name = "llvm.round.f32", fir.runtime}
! PRECISE-DAG: func.func private @llvm.round.f64(f64) -> f64 attributes {fir.bindc_name = "llvm.round.f64", fir.runtime}
! PRECISE-DAG: func.func private @llvm.round.f80(f80) -> f80 attributes {fir.bindc_name = "llvm.round.f80", fir.runtime}
