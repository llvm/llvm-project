! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: maskl_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}
subroutine maskl_test(a, b)
  integer :: a
  integer :: b

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskl(a)
  ! CHECK: %[[C__0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[BITS:.*]] = arith.constant 32 : i32
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_VAL]] : i32
  ! CHECK: %[[SHIFT:.*]] = arith.shli %[[C__1]], %[[LEN]] : i32
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_VAL]], %[[C__0]] : i32
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i32
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i32>
end subroutine maskl_test

! CHECK-LABEL: maskl1_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i8>{{.*}}
subroutine maskl1_test(a, b)
  integer :: a
  integer(kind=1) :: b

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskl(a, 1)
  ! CHECK: %[[C__0:.*]] = arith.constant 0 : i8
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i8
  ! CHECK: %[[BITS:.*]] = arith.constant 8 : i8
  ! CHECK: %[[A_CONV:.*]] = fir.convert %[[A_VAL]] : (i32) -> i8
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_CONV]] : i8
  ! CHECK: %[[SHIFT:.*]] = arith.shli %[[C__1]], %[[LEN]] : i8
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_CONV]], %[[C__0]] : i8
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i8
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i8>
end subroutine maskl1_test

! CHECK-LABEL: maskl2_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i16>{{.*}}
subroutine maskl2_test(a, b)
  integer :: a
  integer(kind=2) :: b

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskl(a, 2)
  ! CHECK: %[[C__0:.*]] = arith.constant 0 : i16
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i16
  ! CHECK: %[[BITS:.*]] = arith.constant 16 : i16
  ! CHECK: %[[A_CONV:.*]] = fir.convert %[[A_VAL]] : (i32) -> i16
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_CONV]] : i16
  ! CHECK: %[[SHIFT:.*]] = arith.shli %[[C__1]], %[[LEN]] : i16
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_CONV]], %[[C__0]] : i16
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i16
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i16>
end subroutine maskl2_test

! CHECK-LABEL: maskl4_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}
subroutine maskl4_test(a, b)
  integer :: a
  integer(kind=4) :: b

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskl(a, 4)
  ! CHECK: %[[C__0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[BITS:.*]] = arith.constant 32 : i32
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_VAL]] : i32
  ! CHECK: %[[SHIFT:.*]] = arith.shli %[[C__1]], %[[LEN]] : i32
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_VAL]], %[[C__0]] : i32
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i32
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i32>
end subroutine maskl4_test

! CHECK-LABEL: maskl8_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i64>{{.*}}
subroutine maskl8_test(a, b)
  integer :: a
  integer(kind=8) :: b

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskl(a, 8)
  ! CHECK: %[[C__0:.*]] = arith.constant 0 : i64
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i64
  ! CHECK: %[[BITS:.*]] = arith.constant 64 : i64
  ! CHECK: %[[A_CONV:.*]] = fir.convert %[[A_VAL]] : (i32) -> i64
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_CONV]] : i64
  ! CHECK: %[[SHIFT:.*]] = arith.shli %[[C__1]], %[[LEN]] : i64
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_CONV]], %[[C__0]] : i64
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i64
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i64>
end subroutine maskl8_test

! TODO: Code containing 128-bit integer literals current breaks. This is
! probably related to the issue linked below. When that is fixed, a test
! for kind=16 should be added here.
!
! https://github.com/llvm/llvm-project/issues/56446
