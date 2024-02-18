! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: merge_bits1_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i8>{{.*}}, %[[JREF:.*]]: !fir.ref<i8>{{.*}}, %[[MREF:.*]]: !fir.ref<i8>{{.*}}, %[[RREF:.*]]: !fir.ref<i8>{{.*}}
subroutine merge_bits1_test(i, j, m, r)
  integer(1) :: i, j, m
  integer(1) :: r

  ! CHECK: %[[I:.*]] = fir.load %[[IREF]] : !fir.ref<i8>
  ! CHECK: %[[J:.*]] = fir.load %[[JREF]] : !fir.ref<i8>
  ! CHECK: %[[M:.*]] = fir.load %[[MREF]] : !fir.ref<i8>
  r = merge_bits(i, j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i8
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i8
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i8
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i8
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i8
  ! CHECK: fir.store %[[RES]] to %[[RREF]] : !fir.ref<i8>
end subroutine merge_bits1_test

! CHECK-LABEL: merge_bits2_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i16>{{.*}}, %[[JREF:.*]]: !fir.ref<i16>{{.*}}, %[[MREF:.*]]: !fir.ref<i16>{{.*}}, %[[RREF:.*]]: !fir.ref<i16>{{.*}}
subroutine merge_bits2_test(i, j, m, r)
  integer(2) :: i, j, m
  integer(2) :: r

  ! CHECK: %[[I:.*]] = fir.load %[[IREF]] : !fir.ref<i16>
  ! CHECK: %[[J:.*]] = fir.load %[[JREF]] : !fir.ref<i16>
  ! CHECK: %[[M:.*]] = fir.load %[[MREF]] : !fir.ref<i16>
  r = merge_bits(i, j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i16
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i16
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i16
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i16
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i16
  ! CHECK: fir.store %[[RES]] to %[[RREF]] : !fir.ref<i16>
end subroutine merge_bits2_test

! CHECK-LABEL: merge_bits4_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i32>{{.*}}, %[[JREF:.*]]: !fir.ref<i32>{{.*}}, %[[MREF:.*]]: !fir.ref<i32>{{.*}}, %[[RREF:.*]]: !fir.ref<i32>{{.*}}
subroutine merge_bits4_test(i, j, m, r)
  integer(4) :: i, j, m
  integer(4) :: r

  ! CHECK: %[[I:.*]] = fir.load %[[IREF]] : !fir.ref<i32>
  ! CHECK: %[[J:.*]] = fir.load %[[JREF]] : !fir.ref<i32>
  ! CHECK: %[[M:.*]] = fir.load %[[MREF]] : !fir.ref<i32>
  r = merge_bits(i, j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i32
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i32
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i32
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i32
  ! CHECK: fir.store %[[RES]] to %[[RREF]] : !fir.ref<i32>
end subroutine merge_bits4_test

! CHECK-LABEL: merge_bits8_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i64>{{.*}}, %[[JREF:.*]]: !fir.ref<i64>{{.*}}, %[[MREF:.*]]: !fir.ref<i64>{{.*}}, %[[RREF:.*]]: !fir.ref<i64>{{.*}}
subroutine merge_bits8_test(i, j, m, r)
  integer(8) :: i, j, m
  integer(8) :: r

  ! CHECK: %[[I:.*]] = fir.load %[[IREF]] : !fir.ref<i64>
  ! CHECK: %[[J:.*]] = fir.load %[[JREF]] : !fir.ref<i64>
  ! CHECK: %[[M:.*]] = fir.load %[[MREF]] : !fir.ref<i64>
  r = merge_bits(i, j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i64
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i64
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i64
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i64
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i64
  ! CHECK: fir.store %[[RES]] to %[[RREF]] : !fir.ref<i64>
end subroutine merge_bits8_test

! CHECK-LABEL: merge_bitsz0_test
! CHECK-SAME: %[[JREF:.*]]: !fir.ref<i32>{{.*}}, %[[MREF:.*]]: !fir.ref<i32>{{.*}}, %[[RREF:.*]]: !fir.ref<i32>{{.*}}
subroutine merge_bitsz0_test(j, m, r)
  integer :: j, m
  integer :: r

  ! CHECK: %[[I:.*]] = arith.constant 13 : i32
  ! CHECK: %[[J:.*]] = fir.load %[[JREF]] : !fir.ref<i32>
  ! CHECK: %[[M:.*]] = fir.load %[[MREF]] : !fir.ref<i32>
  r = merge_bits(B'1101', j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i32
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i32
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i32
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i32
  ! CHECK: fir.store %[[RES]] to %[[RREF]] : !fir.ref<i32>
end subroutine merge_bitsz0_test

! CHECK-LABEL: merge_bitsz1_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i32>{{.*}}, %[[MREF:.*]]: !fir.ref<i32>{{.*}}, %[[RREF:.*]]: !fir.ref<i32>{{.*}}
subroutine merge_bitsz1_test(i, m, r)
  integer :: i, m
  integer :: r

  ! CHECK: %[[I:.*]] = fir.load %[[IREF]] : !fir.ref<i32>
  ! CHECK: %[[J:.*]] = arith.constant 13 : i32
  ! CHECK: %[[M:.*]] = fir.load %[[MREF]] : !fir.ref<i32>
  r = merge_bits(i, Z'0D', m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i32
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i32
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i32
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i32
  ! CHECK: fir.store %[[RES]] to %[[RREF]] : !fir.ref<i32>
end subroutine merge_bitsz1_test
