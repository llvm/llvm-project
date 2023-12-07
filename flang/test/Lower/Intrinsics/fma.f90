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
