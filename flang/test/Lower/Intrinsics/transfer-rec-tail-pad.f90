! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-hlfir %s -o - | FileCheck %s
! REQUIRES: x86-registered-target

! Regression test for https://github.com/llvm/llvm-project/pull/220377
!
! The TRANSFER inline gate compares store sizes, not allocation sizes.
! Tail padding must not be included in the comparison.
!
! The full file lowers to LLVM IR with the intrinsic module path used by lit.
! The module-scoped t2 routine still emits fir.embox; the LLVM lowering path
! handles its descriptor in this test.

! Shape 1: tail-padded record whose store size matches real(10).
!   t1 fields: integer(8) [8B, align 8] + integer(2) [2B, align 2]
!   Store size = 10B; allocation size = alignTo(10, 8) = 16B.
!   real(10) = f80 on x86-64: store size = 10B.
!   10 == 10 on store size -> must INLINE (fir.load), not call _FortranATransfer.
!   Without the storeSizeOnly fix the allocation size (16) != 10 and this
!   would wrongly fall through to the runtime path.
module m1
  type :: t1
    integer(8) :: a
    integer(2) :: b
  end type
end module

! Shape 2: tail-padded record whose allocation size matches integer(8) but
!   whose store size does not.
!   t2 fields: integer(4) [4B, align 4] + integer(1) [1B, align 1]
!   Store size = 5B; allocation size = alignTo(5, 4) = 8B.
!   integer(8) store size = 8B.
!   Without the storeSizeOnly fix: 8 == 8 on allocation size -> would inline,
!   but that reads 3 bytes of uninitialized tail padding into the result.
!   With the fix: 5 != 8 on store size -> correctly stays on RUNTIME path.
module m2
  type :: t2
    integer(4) :: a
    integer(1) :: b
  end type
end module

subroutine transfer_rec_to_real10(out)
  ! CHECK-LABEL: func @_QPtransfer_rec_to_real10(
  ! CHECK-NOT:     fir.call @_FortranATransfer
  ! CHECK:         %[[TMP:.*]] = fir.alloca f80
  ! CHECK:         fir.convert {{.*}} : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<!fir.array<10xi8>>
  ! CHECK:         fir.convert {{.*}} : (!fir.ref<f80>) -> !fir.ref<!fir.array<10xi8>>
  ! CHECK:         fir.copy {{.*}} to {{.*}} no_overlap : !fir.ref<!fir.array<10xi8>>, !fir.ref<!fir.array<10xi8>>
  ! CHECK:         fir.load {{.*}} : !fir.ref<f80>
  ! CHECK:         return
  use m1
  type(t1) :: src
  real(10) :: out
  src%a = 42
  src%b = 7
  out = transfer(src, out)
end subroutine

subroutine transfer_rec_to_int8(res)
  ! CHECK-LABEL: func @_QPtransfer_rec_to_int8(
  ! CHECK:         fir.call @_FortranATransfer
  ! CHECK-NOT:     fir.load {{.*}} : !fir.ref<i64>
  ! CHECK:         return
  use m2
  type(t2) :: x
  integer(8) :: res
  x%a = 1
  x%b = 2_1
  res = transfer(x, res)
end subroutine
! A BIND(C) record may contain internal padding.  TRANSFER must copy the
! physical bytes rather than loading and storing the record aggregate, since
! the latter can replace padding bytes with undef.
subroutine transfer_bindc_record_to_int128(res)
  ! CHECK-LABEL: func @_QPtransfer_bindc_record_to_int128(
  ! CHECK:         %[[TMP128:.*]] = fir.alloca i128
  ! CHECK:         fir.convert {{.*}} : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<!fir.array<16xi8>>
  ! CHECK:         fir.convert {{.*}} : (!fir.ref<i128>) -> !fir.ref<!fir.array<16xi8>>
  ! CHECK:         fir.copy {{.*}} to {{.*}} no_overlap : !fir.ref<!fir.array<16xi8>>, !fir.ref<!fir.array<16xi8>>
  ! CHECK:         fir.load {{.*}} : !fir.ref<i128>
  use iso_c_binding, only: c_int8_t, c_int64_t, c_loc, c_f_pointer
  type, bind(c) :: t
    integer(c_int8_t) :: first
    integer(c_int64_t) :: rest
  end type
  type(t), target :: source
  integer(16), target :: res
  integer(c_int8_t), target :: bytes(16) = [ &
      1_c_int8_t, 2_c_int8_t, 3_c_int8_t, 4_c_int8_t, &
      5_c_int8_t, 6_c_int8_t, 7_c_int8_t, 8_c_int8_t, &
      9_c_int8_t, 10_c_int8_t, 11_c_int8_t, 12_c_int8_t, &
      13_c_int8_t, 14_c_int8_t, 15_c_int8_t, 16_c_int8_t]
  integer(c_int8_t), pointer :: source_bytes(:), result_bytes(:)
  ! Initialize all 16 bytes, including the seven internal padding bytes.
  call c_f_pointer(c_loc(source), source_bytes, [16])
  source_bytes = bytes
  res = transfer(source, res)
  call c_f_pointer(c_loc(res), result_bytes, [16])
  if (any(result_bytes /= bytes)) error stop 'TRANSFER changed bytes'
end subroutine
