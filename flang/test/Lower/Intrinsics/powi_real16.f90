! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefix=CHECK-FAST
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s --check-prefix=CHECK-PRECISE
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefix=CHECK-FAST

! CHECK-PRECISE: fir.call @_FortranAFPow16i({{.*}}){{.*}}: (f128, i32) -> f128
! CHECK-PRECISE: fir.call @_FortranAFPow16i({{.*}}){{.*}}: (f128, i32) -> f128
! CHECK-PRECISE: fir.call @_FortranAFPow16i({{.*}}){{.*}}: (f128, i32) -> f128
! CHECK-PRECISE: fir.call @_FortranAFPow16k({{.*}}){{.*}}: (f128, i64) -> f128
! CHECK-FAST: math.fpowi {{.*}} : f128, i32
! CHECK-FAST: math.fpowi {{.*}} : f128, i32
! CHECK-FAST: math.fpowi {{.*}} : f128, i32
! CHECK-FAST: math.fpowi {{.*}} : f128, i64
  real(16) :: a
  integer(1) :: e1
  integer(2) :: e2
  integer(4) :: e3
  integer(8) :: e4
  a = a ** e1
  a = a ** e2
  a = a ** e3
  a = a ** e4
end
