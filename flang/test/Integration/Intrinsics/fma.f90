!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefix=CHECK-FIR %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-LLVMIR %s

function test_real4(a, x, y)
  use ieee_arithmetic, only: ieee_fma
  real :: a, x, y
  test_real4 = ieee_fma(a, x, y)
end function

! CHECK-LABEL: @_QPtest_real4
! CHECK-FIR: {{%[A-Za-z0-9._]+}} = math.fma {{%[0-9]+}}, {{%[0-9]+}}, {{%[0-9]+}} {{.*}} : f32
! CHECK-LLVMIR: {{%[A-Za-z0-9._]+}} = call {{.*}} float @llvm.fma.f32(float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}})

function test_real8(a, x, y)
  use ieee_arithmetic, only: ieee_fma
  real(8) :: a, x, y
  test_real8 = ieee_fma(a, x, y)
end function

! CHECK-LABEL: @_QPtest_real8
! CHECK-FIR: {{%[A-Za-z0-9._]+}} = math.fma {{%[0-9]+}}, {{%[0-9]+}}, {{%[0-9]+}} {{.*}} : f64
! CHECK-LLVMIR: {{%[A-Za-z0-9._]+}} = call {{.*}} double @llvm.fma.f64(double {{%[0-9]+}}, double {{%[0-9]+}}, double {{%[0-9]+}})
