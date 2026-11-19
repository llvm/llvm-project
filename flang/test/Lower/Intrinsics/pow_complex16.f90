! REQUIRES: flang-frontend-supports-f128
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s --check-prefixes="PRECISE"
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! PRECISE: fir.call @_FortranACPowF128({{.*}}){{.*}}: (complex<f128>, complex<f128>) -> complex<f128>
! CHECK: complex.pow %{{.*}}, %{{.*}} fastmath<contract> : complex<f128>
  complex(16) :: a, b
  b = a ** b
end
