! REQUIRES: x86_64-registered-target
! RUN: bbc -target x86_64-unknown-linux-gnu -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -target x86_64-unknown-linux-gnu -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -target x86_64-unknown-linux-gnu -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -target x86_64-unknown-linux-gnu -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -target x86_64-unknown-linux-gnu -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -target x86_64-unknown-linux-gnu -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL %s

function test_real4(x)
  real :: x, test_real4
  test_real4 = aint(x)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.trunc.f32({{%[A-Za-z0-9._]+}}) {{.*}}: (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = aint(x)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.trunc.f64({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> f64

! TODO: wait until fp128 is supported well in llvm.trunc
!function test_real16(x)
!  real(16) :: x, test_real16
!  test_real16 = aint(x)
!end function

! ALL-DAG: func.func private @llvm.trunc.f32(f32) -> f32 attributes {fir.bindc_name = "llvm.trunc.f32", fir.runtime}
! ALL-DAG: func.func private @llvm.trunc.f64(f64) -> f64 attributes {fir.bindc_name = "llvm.trunc.f64", fir.runtime}
