! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL %s

function dtest_real8(x)
  real(8) :: x, dtest_real8
  dtest_real8 = derf(x)
end function

! ALL-LABEL: @_QPdtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @erf({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> f64

function qtest_real16(x)
  real(16) :: x, qtest_real16
  qtest_real16 = qerf(x)
end function

! ALL-LABEL: @_QPqtest_real16
! CHECK: %{{.*}} = fir.call @_FortranAErfF128(%[[a1]]) {{.*}}: (f128) -> f128
