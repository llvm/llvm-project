! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: shifta1_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i8>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i8>{{.*}}
subroutine shifta1_test(a, b, c)
  integer(kind=1) :: a
  integer :: b
  integer(kind=1) :: c

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i8>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = shifta(a, b)
  ! CHECK: %[[C_BITS:.*]] = arith.constant 8 : i8
  ! CHECK: %[[B_CONV:.*]] = fir.convert %[[B_VAL]] : (i32) -> i8
  ! CHECK: %[[SHIFT_IS_BITWIDTH:.*]] = arith.cmpi uge, %[[B_CONV]], %[[C_BITS]] : i8
  ! CHECK: %[[C0:.*]] = arith.constant 0 : i8
  ! CHECK: %[[CM1:.*]] = arith.constant -1 : i8
  ! CHECK: %[[IS_NEG:.*]] = arith.cmpi slt, %[[A_VAL]], %[[C0]] : i8
  ! CHECK: %[[RES:.*]] = arith.select %[[IS_NEG]], %[[CM1]], %[[C0]] : i8
  ! CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[A_VAL]], %[[B_CONV]] : i8
  ! CHECK: %{{.*}} = arith.select %[[SHIFT_IS_BITWIDTH]], %[[RES]], %[[SHIFTED]] : i8
end subroutine shifta1_test

! CHECK-LABEL: shifta2_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i16>{{.*}}
subroutine shifta2_test(a, b, c)
  integer(kind=2) :: a
  integer :: b
  integer(kind=2) :: c

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i16>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = shifta(a, b)
  ! CHECK: %[[C_BITS:.*]] = arith.constant 16 : i16
  ! CHECK: %[[B_CONV:.*]] = fir.convert %[[B_VAL]] : (i32) -> i16
  ! CHECK: %[[SHIFT_IS_BITWIDTH:.*]] = arith.cmpi uge, %[[B_CONV]], %[[C_BITS]] : i16
  ! CHECK: %[[C0:.*]] = arith.constant 0 : i16
  ! CHECK: %[[CM1:.*]] = arith.constant -1 : i16
  ! CHECK: %[[IS_NEG:.*]] = arith.cmpi slt, %[[A_VAL]], %[[C0]] : i16
  ! CHECK: %[[RES:.*]] = arith.select %[[IS_NEG]], %[[CM1]], %[[C0]] : i16
  ! CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[A_VAL]], %[[B_CONV]] : i16
  ! CHECK: %{{.*}} = arith.select %[[SHIFT_IS_BITWIDTH]], %[[RES]], %[[SHIFTED]] : i16
end subroutine shifta2_test

! CHECK-LABEL: shifta4_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i32>{{.*}}
subroutine shifta4_test(a, b, c)
  integer(kind=4) :: a
  integer :: b
  integer(kind=4) :: c

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = shifta(a, b)
  ! CHECK: %[[C_BITS:.*]] = arith.constant 32 : i32
  ! CHECK: %[[SHIFT_IS_BITWIDTH:.*]] = arith.cmpi uge, %[[B_VAL]], %[[C_BITS]] : i32
  ! CHECK: %[[C0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[CM1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[IS_NEG:.*]] = arith.cmpi slt, %[[A_VAL]], %[[C0]] : i32
  ! CHECK: %[[RES:.*]] = arith.select %[[IS_NEG]], %[[CM1]], %[[C0]] : i32
  ! CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[A_VAL]], %[[B_VAL]] : i32
  ! CHECK: %{{.*}} = arith.select %[[SHIFT_IS_BITWIDTH]], %[[RES]], %[[SHIFTED]] : i32
end subroutine shifta4_test

! CHECK-LABEL: shifta8_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i64>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i64>{{.*}}
subroutine shifta8_test(a, b, c)
  integer(kind=8) :: a
  integer :: b
  integer(kind=8) :: c

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i64>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = shifta(a, b)
  ! CHECK: %[[C_BITS:.*]] = arith.constant 64 : i64
  ! CHECK: %[[B_CONV:.*]] = fir.convert %[[B_VAL]] : (i32) -> i64
  ! CHECK: %[[SHIFT_IS_BITWIDTH:.*]] = arith.cmpi uge, %[[B_CONV]], %[[C_BITS]] : i64
  ! CHECK: %[[C0:.*]] = arith.constant 0 : i64
  ! CHECK: %[[CM1:.*]] = arith.constant -1 : i64
  ! CHECK: %[[IS_NEG:.*]] = arith.cmpi slt, %[[A_VAL]], %[[C0]] : i64
  ! CHECK: %[[RES:.*]] = arith.select %[[IS_NEG]], %[[CM1]], %[[C0]] : i64
  ! CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[A_VAL]], %[[B_CONV]] : i64
  ! CHECK: %{{.*}} = arith.select %[[SHIFT_IS_BITWIDTH]], %[[RES]], %[[SHIFTED]] : i64
end subroutine shifta8_test

! CHECK-LABEL: shifta16_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i128>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i128>{{.*}}
subroutine shifta16_test(a, b, c)
  integer(kind=16) :: a
  integer :: b
  integer(kind=16) :: c

  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i128>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = shifta(a, b)
  ! CHECK: %[[C_BITS:.*]] = arith.constant 128 : i128
  ! CHECK: %[[B_CONV:.*]] = fir.convert %[[B_VAL]] : (i32) -> i128
  ! CHECK: %[[SHIFT_IS_BITWIDTH:.*]] = arith.cmpi uge, %[[B_CONV]], %[[C_BITS]] : i128
  ! CHECK: %[[C0:.*]] = arith.constant 0 : i128
  ! CHECK: %[[CM1:.*]] = arith.constant {{.*}} : i128
  ! CHECK: %[[IS_NEG:.*]] = arith.cmpi slt, %[[A_VAL]], %[[C0]] : i128
  ! CHECK: %[[RES:.*]] = arith.select %[[IS_NEG]], %[[CM1]], %[[C0]] : i128
  ! CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[A_VAL]], %[[B_CONV]] : i128
  ! CHECK: %{{.*}} = arith.select %[[SHIFT_IS_BITWIDTH]], %[[RES]], %[[SHIFTED]] : i128
end subroutine shifta16_test
