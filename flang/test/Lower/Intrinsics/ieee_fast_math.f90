! The IEEE_ARITHMETIC and IEEE_EXCEPTIONS procedures are expansions that encode
! NaN, infinity, and signed zero behavior explicitly. The relaxed floating point
! assumptions requested for the surrounding code must not reach the operations
! implementing them. Contraction is the one flag that still applies, since none
! of these expansions contain contractable arithmetic.

! RUN: %flang_fc1 -emit-fir -ffast-math %s -o %t.fir
! RUN: FileCheck %s --input-file=%t.fir
! RUN: FileCheck %s --check-prefix=NOFAST --input-file=%t.fir
! RUN: %flang_fc1 -emit-fir -menable-no-nans -menable-no-infs -fno-signed-zeros \
! RUN:   -mreassociate -fapprox-func -freciprocal-math -ffp-contract=off %s -o - \
! RUN:   | FileCheck %s --check-prefix=NOFMF
! RUN: %flang_fc1 -emit-llvm -O3 -ffast-math %s -o - | FileCheck %s --check-prefix=LLVM

! Ordinary arithmetic in the same file still gets everything that was asked for.
! CHECK-LABEL: func.func @_QPplain_arith(
subroutine plain_arith(x, y, r)
  real(4) :: x, y, r
  ! CHECK: arith.mulf %{{[^ ]*}}, %{{[^ ]*}} fastmath<fast> : f32
  r = x * y
end subroutine

! CHECK-LABEL: func.func @_QPmin_num_mag(
! NOFAST-LABEL: func.func @_QPmin_num_mag(
! NOFAST-NOT:   fastmath<{{.*(nnan|ninf|nsz|arcp|afn|reassoc|fast).*}}>
! NOFMF-LABEL: func.func @_QPmin_num_mag(
! NOFMF-NOT:   fastmath
! LLVM-LABEL: define {{.*}} @min_num_mag_(
subroutine min_num_mag(x, y, r)
  use ieee_arithmetic
  real(4) :: x, y, r
  ! CHECK: %[[X:.*]] = fir.load %{{[^ ]*}} : !fir.ref<f32>
  ! CHECK: %[[Y:.*]] = fir.load %{{[^ ]*}} : !fir.ref<f32>
  ! CHECK: %[[AX:.*]] = math.copysign %[[X]], %{{[^ ]*}} fastmath<contract> : f32
  ! CHECK: %[[AY:.*]] = math.copysign %[[Y]], %{{[^ ]*}} fastmath<contract> : f32
  ! CHECK: arith.cmpf olt, %[[AX]], %[[AY]] fastmath<contract> : f32
  ! CHECK: arith.cmpf ogt, %[[AX]], %[[AY]] fastmath<contract> : f32
  ! CHECK: arith.cmpf oeq, %[[AX]], %[[AY]] fastmath<contract> : f32
  ! These two ordered compares select the non-NaN operand. Under nnan they fold
  ! to true and the NaN arm of the expansion disappears.
  ! CHECK: arith.cmpf ord, %[[X]], %[[X]] fastmath<contract> : f32
  ! CHECK: arith.cmpf ord, %[[Y]], %[[Y]] fastmath<contract> : f32
  ! LLVM-NOT: fcmp {{.*}}fast
  ! LLVM: fcmp contract olt
  ! LLVM: fcmp contract ogt
  ! LLVM: fcmp contract oeq
  ! LLVM: fcmp contract ord
  ! LLVM: fcmp contract ord
  r = ieee_min_num_mag(x, y)
end subroutine

! CHECK-LABEL: func.func @_QPunordered(
! NOFAST-LABEL: func.func @_QPunordered(
! NOFAST-NOT:   fastmath<{{.*(nnan|ninf|nsz|arcp|afn|reassoc|fast).*}}>
! NOFMF-LABEL: func.func @_QPunordered(
! NOFMF-NOT:   fastmath
! LLVM-LABEL: define {{.*}} @unordered_(
subroutine unordered(x, y, l)
  use ieee_arithmetic
  real(4) :: x, y
  logical :: l
  ! Under nnan this unordered compare folds to false.
  ! CHECK: arith.cmpf uno, %{{[^ ]*}}, %{{[^ ]*}} fastmath<contract> : f32
  ! LLVM: fcmp contract uno
  l = ieee_unordered(x, y)
end subroutine

! CHECK-LABEL: func.func @_QPrem_and_rint(
! NOFAST-LABEL: func.func @_QPrem_and_rint(
! NOFAST-NOT:   fastmath<{{.*(nnan|ninf|nsz|arcp|afn|reassoc|fast).*}}>
! NOFMF-LABEL: func.func @_QPrem_and_rint(
! NOFMF-NOT:   fastmath
subroutine rem_and_rint(x, y, r, s)
  use ieee_arithmetic
  real(4) :: x, y, r, s
  ! An afn call to remainderf would be free to use an approximate variant.
  ! CHECK: fir.call @remainderf(%{{[^ ]*}}, %{{[^ ]*}}) fastmath<contract> : (f32, f32) -> f32
  r = ieee_rem(x, y)
  ! CHECK: %[[RINT:.*]] = fir.call @llvm.nearbyint.f32(%[[ARG:[^ ]*]]) fastmath<contract> : (f32) -> f32
  ! CHECK: arith.cmpf one, %[[ARG]], %[[RINT]] fastmath<contract> : f32
  s = ieee_rint(x)
end subroutine
! NOFAST: return
! NOFMF: return
