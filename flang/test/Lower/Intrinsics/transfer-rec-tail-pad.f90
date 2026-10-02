! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-hlfir %s -o - | FileCheck %s
! REQUIRES: x86-registered-target

! Regression test for https://github.com/llvm/llvm-project/pull/220377
!
! The TRANSFER inline gate compares sizes from getTypeSizeAndAlignment.
! For derived types, this includes tail padding, matching the allocation extent
! used by STORAGE_SIZE and the TRANSFER runtime path.
!
! All checks are at the HLFIR level; LLVM IR lowering is covered by transfer-rec-tail-pad-llvm.f90.

! Shape 1: tail-padded record whose allocation size matches integer(8).
!   t1 fields: integer(4) [4B, align 4] + integer(1) [1B, align 1]
!   Store size = 5B; allocation size = alignTo(5, 4) = 8B.
!   integer(8) store size = 8B, align = 8B.
!   8 == 8 on allocation size -> INLINES via fir.copy into an 8-byte aligned alloca.
!   fir.copy copies the full 8 bytes (including 3 tail-padding bytes), matching
!   the runtime copy width and preserving physical representation.
module m1
  type :: t1
    integer(4) :: a
    integer(1) :: b
  end type
end module

! Shape 2: tail-padded record whose allocation size (16B) does not match real(10) store size (10B).
!   t2 fields: integer(8) [8B, align 8] + integer(2) [2B, align 2]
!   Store size = 10B; allocation size = alignTo(10, 8) = 16B.
!   real(10) = f80 on x86-64: store size = 10B, align = 16B.
!   16 != 10 on size comparison -> stays on RUNTIME path (_FortranATransfer).
module m2
  type :: t2
    integer(8) :: a
    integer(2) :: b
  end type
end module

subroutine transfer_rec_to_int8(res)
  ! CHECK-LABEL: func @_QPtransfer_rec_to_int8(
  ! CHECK-NOT:     fir.call @_FortranATransfer
  ! CHECK:         %[[TMP:.*]] = fir.alloca i64
  ! CHECK:         %[[SRC_BYTES:.*]] = fir.convert {{.*}} : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<!fir.array<8xi8>>
  ! CHECK:         %[[DST_BYTES:.*]] = fir.convert %[[TMP]] : (!fir.ref<i64>) -> !fir.ref<!fir.array<8xi8>>
  ! CHECK:         fir.copy %[[SRC_BYTES]] to %[[DST_BYTES]] no_overlap : !fir.ref<!fir.array<8xi8>>, !fir.ref<!fir.array<8xi8>>
  ! CHECK:         fir.load %[[TMP]] : !fir.ref<i64>
  ! CHECK:         return
  use m1
  type(t1) :: x
  integer(8) :: res
  x%a = 1
  x%b = 2_1
  res = transfer(x, res)
end subroutine

subroutine transfer_rec_to_real10(out)
  ! CHECK-LABEL: func @_QPtransfer_rec_to_real10(
  ! CHECK:         fir.call @_FortranATransfer
  ! CHECK-NOT:     fir.load {{.*}} : !fir.ref<f80>
  ! CHECK:         return
  use m2
  type(t2) :: src
  real(10) :: out
  src%a = 42
  src%b = 7
  out = transfer(src, out)
end subroutine

! A BIND(C) record may contain internal padding.  TRANSFER must copy the
! physical bytes rather than loading and storing the record aggregate, since
! the latter can replace padding bytes with undef.
! This test covers HLFIR data flow only: the RUN line emits HLFIR and does
! not execute the subroutine, so correctness of the byte values is not
! verified here.
subroutine transfer_bindc_record_to_int128(res)
  ! CHECK-LABEL: func @_QPtransfer_bindc_record_to_int128(
  ! CHECK:         %[[TMP128:.*]] = fir.alloca i128
  ! CHECK:         %[[SRC_BYTES128:.*]] = fir.convert {{.*}} : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<!fir.array<16xi8>>
  ! CHECK:         %[[DST_BYTES128:.*]] = fir.convert %[[TMP128]] : (!fir.ref<i128>) -> !fir.ref<!fir.array<16xi8>>
  ! CHECK:         fir.copy %[[SRC_BYTES128]] to %[[DST_BYTES128]] no_overlap : !fir.ref<!fir.array<16xi8>>, !fir.ref<!fir.array<16xi8>>
  ! CHECK:         fir.load %[[TMP128]] : !fir.ref<i128>
  use iso_c_binding, only: c_int8_t, c_int64_t
  type, bind(c) :: t
    integer(c_int8_t) :: first
    integer(c_int64_t) :: rest
  end type
  type(t) :: source
  integer(16) :: res
  res = transfer(source, res)
end subroutine
