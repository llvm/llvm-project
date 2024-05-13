! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: fir.call @_FortranAcqpowk({{.*}}){{.*}}: (!fir.complex<16>, i64) -> !fir.complex<16>
  complex(16) :: a
  integer(8) :: b
  b = a ** b
end
