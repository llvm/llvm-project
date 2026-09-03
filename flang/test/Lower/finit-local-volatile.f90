! Tests that -finit-local= byte-fill loops correctly handle volatile locals.
! The byte-seq ref type and all CoordinateOp result types must carry the
! source volatility flag so that final stores are emitted as "store volatile".
! LLVM IR-level volatile store checks are in finit-local-volatile-llvm.f90.
!
! When source and target types already have matching volatility,
! createConvertWithVolatileCast emits a plain fir.convert (no fir.volatile_cast
! needed).  The strict FIR volatile verifier accepts this because both sides
! carry the volatile flag.  Before the fix, the target type was non-volatile
! so the verifier rejected the convert with "mismatched volatility".
!
! Four paths are covered:
!   1. volatile derived-type local  -- record byte-fill loop in initAddr.
!   2. volatile real(10) local      -- allocation-gap byte-fill loop in
!      genInitLocalStore (requires x86; see finit-local-volatile-real10.f90).
!   3. volatile logical local (hex) -- LOGICAL bitcast address convert in
!      genInitLocalStore must preserve volatility.
!   4. volatile integer array local -- rank-1 array-view convert in initAddr
!      must preserve volatility (both zero and hex modes).
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
! ZERO:        fir.convert %[[V]]#0 : (!fir.ref<{{.*}}, volatile>) -> !fir.ref<!fir.array<?xi8>, volatile>
! ZERO:        fir.do_loop
! ZERO:        fir.store {{.*}} : !fir.ref<i8, volatile>

! HEX-LABEL:  func.func @_QPtest_volatile_derived(
! HEX:         %[[V:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_derivedEv
! HEX:         fir.convert %[[V]]#0 : (!fir.ref<{{.*}}, volatile>) -> !fir.ref<!fir.array<?xi8>, volatile>
! HEX:         fir.do_loop
! HEX:         fir.store {{.*}} : !fir.ref<i8, volatile>

! ---------------------------------------------------------------------------
! Volatile logical local (hex mode) -- LOGICAL bitcast path.
! The bitcast convert preserves volatility: !fir.ref<i32, volatile>.
! ---------------------------------------------------------------------------
subroutine test_volatile_logical(res)
  logical(4), volatile :: l
  integer :: res
  if (l) res = 1
end subroutine

! HEX-LABEL: func.func @_QPtest_volatile_logical(
! HEX:        %[[L:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_logicalEl
! HEX:        fir.convert %[[L]]#0 : (!fir.ref<!fir.logical<4>, volatile>) -> !fir.ref<i32, volatile>
! HEX:        fir.store {{.*}} : !fir.ref<i32, volatile>

! ---------------------------------------------------------------------------
! Volatile integer array local -- rank-1 array-view path.
! The rank-1 view convert preserves volatility end-to-end.
! ---------------------------------------------------------------------------
subroutine test_volatile_array(res)
  integer(4), volatile :: x(4)
  integer :: res
  res = x(1)
end subroutine

! ZERO-LABEL: func.func @_QPtest_volatile_array(
! ZERO:        %[[X:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_arrayEx
! ZERO:        fir.convert %[[X]]#0 : (!fir.ref<!fir.array<4xi32>, volatile>) -> !fir.ref<!fir.array<?xi32>, volatile>
! ZERO:        fir.do_loop
! ZERO:        fir.store {{.*}} : !fir.ref<i32, volatile>

! HEX-LABEL:  func.func @_QPtest_volatile_array(
! HEX:         %[[X:.*]]:2 = hlfir.declare {{.*}}_QFtest_volatile_arrayEx
! HEX:         fir.convert %[[X]]#0 : (!fir.ref<!fir.array<4xi32>, volatile>) -> !fir.ref<!fir.array<?xi32>, volatile>
! HEX:         fir.do_loop
! HEX:         fir.store {{.*}} : !fir.ref<i32, volatile>
