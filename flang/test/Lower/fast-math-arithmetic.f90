! RUN: %flang_fc1 -emit-fir -ffp-contract=fast %s -o - 2>&1 | FileCheck --check-prefixes=CONTRACT,ALL %s
! RUN: %flang_fc1 -emit-fir -menable-no-infs %s -o - 2>&1 | FileCheck --check-prefixes=NINF,ALL %s

! ALL-LABEL: func.func @_QPtest
subroutine test(x)
  real x
! CONTRACT: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:contract]]> : f32
! NINF: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:ninf]]> : f32
! ALL: arith.divf{{.*}}, {{.*}} fastmath<[[ATTRS]]> : f32
! ALL: arith.addf{{.*}}, {{.*}} fastmath<[[ATTRS]]> : f32
! ALL: arith.subf{{.*}}, {{.*}} fastmath<[[ATTRS]]> : f32
  x = x * x + x / x - x
end subroutine test
