! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! REQUIRES: x86-registered-target

! Regression test for https://github.com/llvm/llvm-project/pull/220377
!
! The source record has a 10-byte stored representation and 16-byte allocation
! extent.  TRANSFER to real(10) must copy the 10 bytes into result-aligned
! storage before loading the result.
module m
  type :: t
    integer(8) :: a
    integer(2) :: b
  end type
end module

subroutine transfer_rec_to_real10(out)
  use m
  type(t) :: src
  real(10) :: out
  src%a = 42
  src%b = 7
  out = transfer(src, out)
end subroutine

! CHECK-LABEL: define{{.*}} @transfer_rec_to_real10_(
! CHECK:       %[[TMP:.*]] = alloca x86_fp80
! CHECK:       call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 10, i1 false)
! CHECK:       %[[RESULT:.*]] = load x86_fp80, ptr %[[TMP]], align 16
