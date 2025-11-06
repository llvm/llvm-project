! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: maskr_test
subroutine maskr_test(a, b)
  integer :: a
  integer :: b
  ! CHECK-DAG: %[[BITS:.*]] = arith.constant 32 : i32
  ! CHECK-DAG: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK-DAG: %[[C__0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[A:.*]] = fir.declare %{{.*}}Ea
  ! CHECK: %[[B:.*]] = fir.declare %{{.*}}Eb

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskr(a)
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_VAL]] : i32
  ! CHECK: %[[SHIFT:.*]] = arith.shrui %[[C__1]], %[[LEN]] : i32
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_VAL]], %[[C__0]] : i32
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i32
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i32>
end subroutine maskr_test

! CHECK-LABEL: maskr1_test
subroutine maskr1_test(a, b)
  integer :: a
  integer(kind=1) :: b
  ! CHECK-DAG: %[[BITS:.*]] = arith.constant 8 : i8
  ! CHECK-DAG: %[[C__1:.*]] = arith.constant -1 : i8
  ! CHECK-DAG: %[[C__0:.*]] = arith.constant 0 : i8
  ! CHECK: %[[A:.*]] = fir.declare %{{.*}}Ea
  ! CHECK: %[[B:.*]] = fir.declare %{{.*}}Eb

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskr(a, 1)
  ! CHECK: %[[A_CONV:.*]] = fir.convert %[[A_VAL]] : (i32) -> i8
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_CONV]] : i8
  ! CHECK: %[[SHIFT:.*]] = arith.shrui %[[C__1]], %[[LEN]] : i8
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_CONV]], %[[C__0]] : i8
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i8
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i8>
end subroutine maskr1_test

! CHECK-LABEL: maskr2_test
subroutine maskr2_test(a, b)
  integer :: a
  integer(kind=2) :: b
  ! CHECK-DAG: %[[BITS:.*]] = arith.constant 16 : i16
  ! CHECK-DAG: %[[C__1:.*]] = arith.constant -1 : i16
  ! CHECK-DAG: %[[C__0:.*]] = arith.constant 0 : i16
  ! CHECK: %[[A:.*]] = fir.declare %{{.*}}Ea
  ! CHECK: %[[B:.*]] = fir.declare %{{.*}}Eb

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskr(a, 2)
  ! CHECK: %[[A_CONV:.*]] = fir.convert %[[A_VAL]] : (i32) -> i16
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_CONV]] : i16
  ! CHECK: %[[SHIFT:.*]] = arith.shrui %[[C__1]], %[[LEN]] : i16
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_CONV]], %[[C__0]] : i16
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i16
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i16>
end subroutine maskr2_test

! CHECK-LABEL: maskr4_test
subroutine maskr4_test(a, b)
  integer :: a
  integer(kind=4) :: b
  ! CHECK-DAG: %[[BITS:.*]] = arith.constant 32 : i32
  ! CHECK-DAG: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK-DAG: %[[C__0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[A:.*]] = fir.declare %{{.*}}Ea
  ! CHECK: %[[B:.*]] = fir.declare %{{.*}}Eb

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskr(a, 4)
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_VAL]] : i32
  ! CHECK: %[[SHIFT:.*]] = arith.shrui %[[C__1]], %[[LEN]] : i32
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_VAL]], %[[C__0]] : i32
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i32
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i32>
end subroutine maskr4_test

! CHECK-LABEL: maskr8_test
subroutine maskr8_test(a, b)
  integer :: a
  integer(kind=8) :: b
  ! CHECK-DAG: %[[BITS:.*]] = arith.constant 64 : i64
  ! CHECK-DAG: %[[C__1:.*]] = arith.constant -1 : i64
  ! CHECK-DAG: %[[C__0:.*]] = arith.constant 0 : i64
  ! CHECK: %[[A:.*]] = fir.declare %{{.*}}Ea
  ! CHECK: %[[B:.*]] = fir.declare %{{.*}}Eb

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskr(a, 8)
  ! CHECK: %[[A_CONV:.*]] = fir.convert %[[A_VAL]] : (i32) -> i64
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_CONV]] : i64
  ! CHECK: %[[SHIFT:.*]] = arith.shrui %[[C__1]], %[[LEN]] : i64
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_CONV]], %[[C__0]] : i64
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i64
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i64>
end subroutine maskr8_test

subroutine maskr16_test(a, b)
  integer :: a
  integer(16) :: b
  ! CHECK-DAG: %[[BITS:.*]] = arith.constant 128 : i128
  ! CHECK-DAG: %[[C__1:.*]] = arith.constant -1 : i128
  ! CHECK-DAG: %[[C__0:.*]] = arith.constant 0 : i128
  ! CHECK: %[[A:.*]] = fir.declare %{{.*}}Ea
  ! CHECK: %[[B:.*]] = fir.declare %{{.*}}Eb

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  b = maskr(a, 16)
  ! CHECK: %[[A_CONV:.*]] = fir.convert %[[A_VAL]] : (i32) -> i128
  ! CHECK: %[[LEN:.*]] = arith.subi %[[BITS]], %[[A_CONV]] : i128
  ! CHECK: %[[SHIFT:.*]] = arith.shrui %[[C__1]], %[[LEN]] : i128
  ! CHECK: %[[IS0:.*]] = arith.cmpi eq, %[[A_CONV]], %[[C__0]] : i128
  ! CHECK: %[[RESULT:.*]] = arith.select %[[IS0]], %[[C__0]], %[[SHIFT]] : i128
  ! CHECK: fir.store %[[RESULT]] to %[[B]] : !fir.ref<i128>
end subroutine
