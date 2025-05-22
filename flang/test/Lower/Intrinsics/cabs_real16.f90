! REQUIRES: flang-supports-f128-math
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: fir.call @_FortranACAbsF128({{.*}}){{.*}}: (!fir.complex<16>) -> f128
  complex(16) :: a
  real(16) :: b
  b = abs(a)
end
