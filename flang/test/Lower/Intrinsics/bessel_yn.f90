! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL %s

function test_real4(x, n)
  real :: x, test_real4
  integer :: n
  test_real4 = bessel_yn(n, x)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @ynf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (i32, f32) -> f32

function test_real8(x, n)
  real(8) :: x, test_real8
  integer :: n
  test_real8 = bessel_yn(n, x)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @yn({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) {{.*}}: (i32, f64) -> f64
