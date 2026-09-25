! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: merge_bits1_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i8>{{.*}}, %[[JREF:.*]]: !fir.ref<i8>{{.*}}, %[[MREF:.*]]: !fir.ref<i8>{{.*}}, %[[RREF:.*]]: !fir.ref<i8>{{.*}}
subroutine merge_bits1_test(i, j, m, r)
  integer(1) :: i, j, m
  integer(1) :: r

  ! CHECK-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %[[IREF]] {{.*}} {uniq_name = "_QFmerge_bits1_testEi"}
  ! CHECK-DAG: %[[J_DECL:.*]]:2 = hlfir.declare %[[JREF]] {{.*}} {uniq_name = "_QFmerge_bits1_testEj"}
  ! CHECK-DAG: %[[M_DECL:.*]]:2 = hlfir.declare %[[MREF]] {{.*}} {uniq_name = "_QFmerge_bits1_testEm"}
  ! CHECK-DAG: %[[R_DECL:.*]]:2 = hlfir.declare %[[RREF]] {{.*}} {uniq_name = "_QFmerge_bits1_testEr"}
  ! CHECK: %[[I:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i8>
  ! CHECK: %[[J:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<i8>
  ! CHECK: %[[M:.*]] = fir.load %[[M_DECL]]#0 : !fir.ref<i8>
  r = merge_bits(i, j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i8
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i8
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i8
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i8
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i8
  ! CHECK: hlfir.assign %[[RES]] to %[[R_DECL]]#0 : i8, !fir.ref<i8>
end subroutine merge_bits1_test

! CHECK-LABEL: merge_bits2_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i16>{{.*}}, %[[JREF:.*]]: !fir.ref<i16>{{.*}}, %[[MREF:.*]]: !fir.ref<i16>{{.*}}, %[[RREF:.*]]: !fir.ref<i16>{{.*}}
subroutine merge_bits2_test(i, j, m, r)
  integer(2) :: i, j, m
  integer(2) :: r

  ! CHECK-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %[[IREF]] {{.*}} {uniq_name = "_QFmerge_bits2_testEi"}
  ! CHECK-DAG: %[[J_DECL:.*]]:2 = hlfir.declare %[[JREF]] {{.*}} {uniq_name = "_QFmerge_bits2_testEj"}
  ! CHECK-DAG: %[[M_DECL:.*]]:2 = hlfir.declare %[[MREF]] {{.*}} {uniq_name = "_QFmerge_bits2_testEm"}
  ! CHECK-DAG: %[[R_DECL:.*]]:2 = hlfir.declare %[[RREF]] {{.*}} {uniq_name = "_QFmerge_bits2_testEr"}
  ! CHECK: %[[I:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i16>
  ! CHECK: %[[J:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<i16>
  ! CHECK: %[[M:.*]] = fir.load %[[M_DECL]]#0 : !fir.ref<i16>
  r = merge_bits(i, j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i16
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i16
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i16
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i16
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i16
  ! CHECK: hlfir.assign %[[RES]] to %[[R_DECL]]#0 : i16, !fir.ref<i16>
end subroutine merge_bits2_test

! CHECK-LABEL: merge_bits4_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i32>{{.*}}, %[[JREF:.*]]: !fir.ref<i32>{{.*}}, %[[MREF:.*]]: !fir.ref<i32>{{.*}}, %[[RREF:.*]]: !fir.ref<i32>{{.*}}
subroutine merge_bits4_test(i, j, m, r)
  integer(4) :: i, j, m
  integer(4) :: r

  ! CHECK-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %[[IREF]] {{.*}} {uniq_name = "_QFmerge_bits4_testEi"}
  ! CHECK-DAG: %[[J_DECL:.*]]:2 = hlfir.declare %[[JREF]] {{.*}} {uniq_name = "_QFmerge_bits4_testEj"}
  ! CHECK-DAG: %[[M_DECL:.*]]:2 = hlfir.declare %[[MREF]] {{.*}} {uniq_name = "_QFmerge_bits4_testEm"}
  ! CHECK-DAG: %[[R_DECL:.*]]:2 = hlfir.declare %[[RREF]] {{.*}} {uniq_name = "_QFmerge_bits4_testEr"}
  ! CHECK: %[[I:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[J:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[M:.*]] = fir.load %[[M_DECL]]#0 : !fir.ref<i32>
  r = merge_bits(i, j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i32
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i32
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i32
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i32
  ! CHECK: hlfir.assign %[[RES]] to %[[R_DECL]]#0 : i32, !fir.ref<i32>
end subroutine merge_bits4_test

! CHECK-LABEL: merge_bits8_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i64>{{.*}}, %[[JREF:.*]]: !fir.ref<i64>{{.*}}, %[[MREF:.*]]: !fir.ref<i64>{{.*}}, %[[RREF:.*]]: !fir.ref<i64>{{.*}}
subroutine merge_bits8_test(i, j, m, r)
  integer(8) :: i, j, m
  integer(8) :: r

  ! CHECK-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %[[IREF]] {{.*}} {uniq_name = "_QFmerge_bits8_testEi"}
  ! CHECK-DAG: %[[J_DECL:.*]]:2 = hlfir.declare %[[JREF]] {{.*}} {uniq_name = "_QFmerge_bits8_testEj"}
  ! CHECK-DAG: %[[M_DECL:.*]]:2 = hlfir.declare %[[MREF]] {{.*}} {uniq_name = "_QFmerge_bits8_testEm"}
  ! CHECK-DAG: %[[R_DECL:.*]]:2 = hlfir.declare %[[RREF]] {{.*}} {uniq_name = "_QFmerge_bits8_testEr"}
  ! CHECK: %[[I:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i64>
  ! CHECK: %[[J:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<i64>
  ! CHECK: %[[M:.*]] = fir.load %[[M_DECL]]#0 : !fir.ref<i64>
  r = merge_bits(i, j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i64
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i64
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i64
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i64
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i64
  ! CHECK: hlfir.assign %[[RES]] to %[[R_DECL]]#0 : i64, !fir.ref<i64>
end subroutine merge_bits8_test

! CHECK-LABEL: merge_bitsz0_test
! CHECK-SAME: %[[JREF:.*]]: !fir.ref<i32>{{.*}}, %[[MREF:.*]]: !fir.ref<i32>{{.*}}, %[[RREF:.*]]: !fir.ref<i32>{{.*}}
subroutine merge_bitsz0_test(j, m, r)
  integer :: j, m
  integer :: r

  ! CHECK-DAG: %[[J_DECL:.*]]:2 = hlfir.declare %[[JREF]] {{.*}} {uniq_name = "_QFmerge_bitsz0_testEj"}
  ! CHECK-DAG: %[[M_DECL:.*]]:2 = hlfir.declare %[[MREF]] {{.*}} {uniq_name = "_QFmerge_bitsz0_testEm"}
  ! CHECK-DAG: %[[R_DECL:.*]]:2 = hlfir.declare %[[RREF]] {{.*}} {uniq_name = "_QFmerge_bitsz0_testEr"}
  ! CHECK: %[[I:.*]] = arith.constant 13 : i32
  ! CHECK: %[[J:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[M:.*]] = fir.load %[[M_DECL]]#0 : !fir.ref<i32>
  r = merge_bits(B'1101', j, m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i32
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i32
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i32
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i32
  ! CHECK: hlfir.assign %[[RES]] to %[[R_DECL]]#0 : i32, !fir.ref<i32>
end subroutine merge_bitsz0_test

! CHECK-LABEL: merge_bitsz1_test
! CHECK-SAME: %[[IREF:.*]]: !fir.ref<i32>{{.*}}, %[[MREF:.*]]: !fir.ref<i32>{{.*}}, %[[RREF:.*]]: !fir.ref<i32>{{.*}}
subroutine merge_bitsz1_test(i, m, r)
  integer :: i, m
  integer :: r

  ! CHECK-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %[[IREF]] {{.*}} {uniq_name = "_QFmerge_bitsz1_testEi"}
  ! CHECK-DAG: %[[M_DECL:.*]]:2 = hlfir.declare %[[MREF]] {{.*}} {uniq_name = "_QFmerge_bitsz1_testEm"}
  ! CHECK-DAG: %[[R_DECL:.*]]:2 = hlfir.declare %[[RREF]] {{.*}} {uniq_name = "_QFmerge_bitsz1_testEr"}
  ! CHECK: %[[J:.*]] = arith.constant 13 : i32
  ! CHECK: %[[I:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[M:.*]] = fir.load %[[M_DECL]]#0 : !fir.ref<i32>
  r = merge_bits(i, Z'0D', m)
  ! CHECK: %[[C__1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[NM:.*]] = arith.xori %[[M]], %[[C__1]] : i32
  ! CHECK: %[[LFT:.*]] = arith.andi %[[I]], %[[M]] : i32
  ! CHECK: %[[RGT:.*]] = arith.andi %[[J]], %[[NM]] : i32
  ! CHECK: %[[RES:.*]] = arith.ori %[[LFT]], %[[RGT]] : i32
  ! CHECK: hlfir.assign %[[RES]] to %[[R_DECL]]#0 : i32, !fir.ref<i32>
end subroutine merge_bitsz1_test
