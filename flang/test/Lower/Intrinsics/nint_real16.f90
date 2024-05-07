! REQUIRES: flang-supports-f128-math
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: fir.call @_FortranALroundF128({{.*}}){{.*}}: (f128) -> i32
! CHECK: fir.call @_FortranALlroundF128({{.*}}){{.*}}: (f128) -> i64
  real(16) :: a
  integer(4) :: b
  integer(8) :: c
  b = nint(a, 4)
  c = nint(a, 8)
end
