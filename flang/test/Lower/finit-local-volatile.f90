! Tests that -finit-local= byte-fill loops correctly handle volatile locals.
! Each byte-view conversion must go through fir.volatile_cast before the
! fir.convert that reinterprets the address as a byte sequence, so that the
! strict FIR volatile verifier does not see a "mismatched volatility" error.
!
! Before the fix both zero and hex modes emitted a plain fir.convert from
! a volatile ref directly to a non-volatile byte-sequence ref, which the
! strict verifier rejects with "mismatched volatility".
!
! Two shapes are covered:
!   1. volatile derived-type local  -- exercises the record byte-fill loop
!      in initAddr (fir.volatile_cast before the i8-array view convert).
!   2. volatile real(10) local      -- exercises the allocation-gap byte-fill
!      loop in genInitLocalStore (same fix, different site).
!      Requires x86 so that real(10) = x86_fp80 with a 6-byte padding gap.
!
! RUN: %flang_fc1 -emit-hlfir -mmlir --strict-fir-volatile-verifier \
! RUN:     -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s
! RUN: %flang_fc1 -emit-hlfir -mmlir --strict-fir-volatile-verifier \
! RUN:     -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX  %s

! ---------------------------------------------------------------------------
! Volatile derived-type local -- record byte-fill loop path.
! ---------------------------------------------------------------------------
subroutine test_volatile_derived(oa)
  type :: t
    integer(4) :: a
    integer(1) :: b
  end type
  type(t), volatile :: v
  integer :: oa
  v%a = 1
  oa = v%a
end subroutine

! ZERO-LABEL: func.func @_QPtest_volatile_derived(
! ZERO:        %[[V:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_derivedEv
! ZERO:        %[[NV:.*]] = fir.volatile_cast %[[V]]#0 : (!fir.ref<{{.*}}, volatile>) -> !fir.ref<{{.*}}>
! ZERO:        fir.convert %[[NV]] : (!fir.ref<{{.*}}>) -> !fir.ref<!fir.array<?xi8>>
! ZERO:        fir.do_loop
! ZERO:        fir.store {{.*}} : !fir.ref<i8>

! HEX-LABEL:  func.func @_QPtest_volatile_derived(
! HEX:         %[[V:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_derivedEv
! HEX:         %[[NV:.*]] = fir.volatile_cast %[[V]]#0 : (!fir.ref<{{.*}}, volatile>) -> !fir.ref<{{.*}}>
! HEX:         fir.convert %[[NV]] : (!fir.ref<{{.*}}>) -> !fir.ref<!fir.array<?xi8>>
! HEX:         fir.do_loop
! HEX:         fir.store {{.*}} : !fir.ref<i8>

! ---------------------------------------------------------------------------
! Volatile real(10) local -- allocation-gap byte-fill loop path (x86 only).
! ---------------------------------------------------------------------------
! REQUIRES: x86-registered-target
subroutine test_volatile_real10(res)
  real(10), volatile :: x
  real(10) :: res
  res = x
end subroutine

! ZERO-LABEL: func.func @_QPtest_volatile_real10(
! ZERO:        %[[X:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_real10Ex
! ZERO:        %[[NV:.*]] = fir.volatile_cast %[[X]]#0 : (!fir.ref<f80, volatile>) -> !fir.ref<f80>
! ZERO:        fir.convert %[[NV]] : (!fir.ref<f80>) -> !fir.ref<!fir.array<?xi8>>
! ZERO:        fir.do_loop
! ZERO:        fir.store {{.*}} : !fir.ref<i8>

! HEX-LABEL:  func.func @_QPtest_volatile_real10(
! HEX:         %[[X:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_real10Ex
! HEX:         %[[NV:.*]] = fir.volatile_cast %[[X]]#0 : (!fir.ref<f80, volatile>) -> !fir.ref<f80>
! HEX:         fir.convert %[[NV]] : (!fir.ref<f80>) -> !fir.ref<!fir.array<?xi8>>
! HEX:         fir.do_loop
! HEX:         fir.store {{.*}} : !fir.ref<i8>
