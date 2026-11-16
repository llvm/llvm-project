! REQUIRES: flang-frontend-supports-f128
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: fir.call @_FortranAY1F128({{.*}}){{.*}}: (f128) -> f128
  real(16) :: a, b
  b = bessel_y1(a)
end
