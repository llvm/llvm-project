! RUN: %flang_fc1 -emit-hlfir -O3 %s -o - | FileCheck %s --check-prefix=CHECK-O3
! RUN: %flang_fc1 -emit-hlfir -O3 -ffast-math %s -o - | FileCheck %s --check-prefix=CHECK-FAST
! RUN: %flang_fc1 -emit-hlfir -O3 -ffast-math -fno-protect-parens %s -o - | FileCheck %s --check-prefix=CHECK-FAST-NO-PROTECT

subroutine test_array_parens
  real, dimension(10) :: a, b, c, d, e
  b = 1.0
  c = 2.0
  d = 3.0
  e = 4.0
  a = b * (c * d * e)
  print *, a
end subroutine

! With -O3, fastmath<contract> everywhere and protect parens
! CHECK-O3: hlfir.elemental
! CHECK-O3: arith.mulf {{.*}} fastmath<contract>
! CHECK-O3: hlfir.elemental
! CHECK-O3: arith.mulf {{.*}} fastmath<contract>
! CHECK-O3: hlfir.elemental
! CHECK-O3: hlfir.no_reassoc
! CHECK-O3: hlfir.elemental
! CHECK-O3: arith.mulf {{.*}} fastmath<contract>

! With -O3 -ffast-math, regulare computations have fastmath<fast>, but still
! protect parens, so the last multiplication is fastmath<contract>
! CHECK-FAST: hlfir.elemental
! CHECK-FAST: arith.mulf {{.*}} fastmath<fast>
! CHECK-FAST: hlfir.elemental
! CHECK-FAST: arith.mulf {{.*}} fastmath<fast>
! CHECK-FAST: hlfir.elemental
! CHECK-FAST: hlfir.no_reassoc
! CHECK-FAST: hlfir.elemental
! CHECK-FAST: arith.mulf {{.*}} fastmath<{{.*}}contract

! With -O3 -ffast-math -fno-protect-parens, fastmath<fast> everywhere
! (don't protect parens)
! CHECK-FAST-NO-PROTECT: hlfir.elemental
! CHECK-FAST-NO-PROTECT: arith.mulf {{.*}} fastmath<fast>
! CHECK-FAST-NO-PROTECT: hlfir.elemental
! CHECK-FAST-NO-PROTECT: arith.mulf {{.*}} fastmath<fast>
! CHECK-FAST-NO-PROTECT: hlfir.elemental
! CHECK-FAST-NO-PROTECT: hlfir.no_reassoc
! CHECK-FAST-NO-PROTECT: hlfir.elemental
! CHECK-FAST-NO-PROTECT: arith.mulf {{.*}} fastmath<{{.*}}fast
