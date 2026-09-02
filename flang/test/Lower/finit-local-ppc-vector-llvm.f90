! Tests that PowerPC vector locals (fir::VectorType) are excluded from
! -finit-local= initialization because fir::VectorType does not implement
! DataLayoutTypeInterface at the HLFIR level.
! The local variable 'x' must not receive any initialization store.
!
! REQUIRES: target=powerpc{{.*}}
!
! RUN: %flang_fc1 -emit-hlfir -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-hlfir -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! vector(real(4)) -- excluded; must not produce any fir.store for 'x'.
subroutine test_vec_real4(res)
  vector(real(4)) :: res
  vector(real(4)) :: x
  res = x
end subroutine

! HEX-LABEL: func.func @_QPtest_vec_real4(
! HEX:        hlfir.declare {{.*}}_QFtest_vec_real4Ex
! HEX-NOT:    fir.store {{.*}} to %{{.*}}#0

! ZERO-LABEL: func.func @_QPtest_vec_real4(
! ZERO:        hlfir.declare {{.*}}_QFtest_vec_real4Ex
! ZERO-NOT:    fir.store {{.*}} to %{{.*}}#0
