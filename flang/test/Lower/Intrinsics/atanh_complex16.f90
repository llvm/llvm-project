! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: fir.call @_FortranACAtanhF128({{.*}}){{.*}}: (!fir.complex<16>) -> !fir.complex<16>
  complex(16) :: a, b
  b = atanh(a)
end
