! Test that -ffast-math uses plain fptosi/fptoui for float-to-integer
! conversions instead of the saturating intrinsics (llvm.fptosi.sat /
! llvm.fptoui.sat).

! RUN: %flang -O2 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=SAFE
! RUN: %flang -O2 -ffast-math -S -emit-llvm %s -o - | FileCheck %s --check-prefix=FAST

! SAFE-LABEL: define {{.*}} @float_to_int_
! SAFE: call i32 @llvm.fptosi.sat.i32.f32
! SAFE-NOT: fptosi float

! FAST-LABEL: define {{.*}} @float_to_int_
! FAST: fptosi float
! FAST-NOT: llvm.fptosi.sat

subroutine float_to_int(x, i)
  real, intent(in) :: x
  integer, intent(out) :: i
  i = x
end subroutine

! SAFE-LABEL: define {{.*}} @double_to_int_
! SAFE: call i32 @llvm.fptosi.sat.i32.f64
! SAFE-NOT: fptosi double

! FAST-LABEL: define {{.*}} @double_to_int_
! FAST: fptosi double
! FAST-NOT: llvm.fptosi.sat

subroutine double_to_int(d, i)
  double precision, intent(in) :: d
  integer, intent(out) :: i
  i = d
end subroutine

! SAFE-LABEL: define {{.*}} @float_to_int8_
! SAFE: call i64 @llvm.fptosi.sat.i64.f32

! FAST-LABEL: define {{.*}} @float_to_int8_
! FAST: fptosi float
! FAST-NOT: llvm.fptosi.sat

subroutine float_to_int8(x, i)
  real, intent(in) :: x
  integer(8), intent(out) :: i
  i = x
end subroutine
