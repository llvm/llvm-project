! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: fir.call @_FortranAFmaF128({{.*}}){{.*}}: (f128, f128, f128) -> f128
  use ieee_arithmetic, only: ieee_fma
  real(16) :: x, y, z
  x = ieee_fma(x, y, z)
end
