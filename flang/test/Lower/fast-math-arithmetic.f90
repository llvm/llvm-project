! RUN: %flang_fc1 -emit-fir -ffp-contract=fast %s -o - 2>&1 | FileCheck --check-prefixes=CONTRACT,ALL %s
! RUN: %flang_fc1 -emit-fir -menable-no-infs %s -o - 2>&1 | FileCheck --check-prefixes=NINF,ALL %s
! RUN: %flang_fc1 -emit-fir -menable-no-nans %s -o - 2>&1 | FileCheck --check-prefixes=NNAN,ALL %s
! RUN: %flang_fc1 -emit-fir -fapprox-func %s -o - 2>&1 | FileCheck --check-prefixes=AFN,ALL %s
! RUN: %flang_fc1 -emit-fir -fno-signed-zeros %s -o - 2>&1 | FileCheck --check-prefixes=NSZ,ALL %s
! RUN: %flang_fc1 -emit-fir -mreassociate %s -o - 2>&1 | FileCheck --check-prefixes=REASSOC,ALL %s
! RUN: %flang_fc1 -emit-fir -freciprocal-math %s -o - 2>&1 | FileCheck --check-prefixes=ARCP,ALL %s
! RUN: %flang_fc1 -emit-fir -ffp-contract=fast -menable-no-infs -menable-no-nans -fapprox-func -fno-signed-zeros -mreassociate -freciprocal-math %s -o - 2>&1 | FileCheck --check-prefixes=FAST,ALL %s

! ALL-LABEL: func.func @_QPtest
subroutine test(x)
  real x
! CONTRACT: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:contract]]> : f32
! NINF: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:ninf]]> : f32
! NNAN: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:nnan]]> : f32
! AFN: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:afn]]> : f32
! NSZ: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:nsz]]> : f32
! REASSOC: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:reassoc]]> : f32
! ARCP: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:arcp]]> : f32
! FAST: arith.mulf{{.*}}, {{.*}} fastmath<[[ATTRS:fast]]> : f32
! ALL: arith.divf{{.*}}, {{.*}} fastmath<[[ATTRS]]> : f32
! ALL: arith.addf{{.*}}, {{.*}} fastmath<[[ATTRS]]> : f32
! ALL: arith.subf{{.*}}, {{.*}} fastmath<[[ATTRS]]> : f32
  x = x * x + x / x - x
end subroutine test
