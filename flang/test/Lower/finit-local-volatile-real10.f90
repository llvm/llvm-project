! Volatile real(10) local -- allocation-gap byte-fill loop path.
! Pinned to x86_64 triple so the test uses x86_fp80 (10-byte store,
! 16-byte allocation) regardless of host architecture.
!
! REQUIRES: x86-registered-target
!
! RUN: %flang_fc1 -emit-hlfir -triple x86_64-unknown-linux-gnu \
! RUN:     -mmlir --strict-fir-volatile-verifier \
! RUN:     -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s
! RUN: %flang_fc1 -emit-hlfir -triple x86_64-unknown-linux-gnu \
! RUN:     -mmlir --strict-fir-volatile-verifier \
! RUN:     -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX  %s

subroutine test_volatile_real10(res)
  real(10), volatile :: x
  real(10) :: res
  res = x
end subroutine

! ZERO-LABEL: func.func @_QPtest_volatile_real10(
! ZERO:        %[[X:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_real10Ex
! ZERO:        fir.convert %[[X]]#0 : (!fir.ref<f80, volatile>) -> !fir.ref<!fir.array<?xi8>, volatile>
! ZERO:        fir.do_loop
! ZERO:        fir.store {{.*}} : !fir.ref<i8, volatile>

! HEX-LABEL:  func.func @_QPtest_volatile_real10(
! HEX:         %[[X:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_real10Ex
! HEX:         fir.convert %[[X]]#0 : (!fir.ref<f80, volatile>) -> !fir.ref<!fir.array<?xi8>, volatile>
! HEX:         fir.do_loop
! HEX:         fir.store {{.*}} : !fir.ref<i8, volatile>
