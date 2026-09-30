! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! REQUIRES: x86-registered-target

! Regression test for https://github.com/llvm/llvm-project/pull/220377
!
! The source record has a 5-byte stored representation and 8-byte allocation
! extent (including 3 bytes of tail padding).  TRANSFER to integer(8) must copy
! all 8 bytes into result-aligned storage before loading the result.
module m
  type :: t
    integer(4) :: a
    integer(1) :: b
  end type
end module

subroutine transfer_rec_to_int8(out)
  use m
  type(t) :: src
  integer(8) :: out
  src%a = 42
  src%b = 7_1
  out = transfer(src, out)
end subroutine

! CHECK-LABEL: define{{.*}} @transfer_rec_to_int8_(
! CHECK-DAG:   %[[TMP:.*]] = alloca i64{{.*}}, align 8
! CHECK-DAG:   %[[SRC:.*]] = alloca %_QMmTt,
! CHECK:       call void @llvm.memcpy.p0.p0.i64(ptr %[[TMP]], ptr %[[SRC]], i64 8, i1 false)
! CHECK:       %[[RESULT:.*]] = load i64, ptr %[[TMP]], align 8
