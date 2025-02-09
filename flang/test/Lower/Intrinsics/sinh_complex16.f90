! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: fir.call @_FortranACSinhF128({{.*}}){{.*}}: (complex<f128>) -> complex<f128>
  complex(16) :: a, b
  b = sinh(a)
end
