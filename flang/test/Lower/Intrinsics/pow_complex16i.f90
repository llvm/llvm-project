! REQUIRES: flang-supports-f128-math
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s --check-prefixes="PRECISE"
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! PRECISE: fir.call @_FortranAcqpowi({{.*}}){{.*}}: (complex<f128>, i32) -> complex<f128>
! CHECK: complex.powi %{{.*}}, %{{.*}} fastmath<contract> : complex<f128>
  complex(16) :: a
  integer(4) :: b
  b = a ** b
end
