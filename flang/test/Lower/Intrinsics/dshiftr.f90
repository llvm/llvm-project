! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdshiftr1_test(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i8>{{.*}}, %[[B:.*]]: !fir.ref<i8>{{.*}}, %[[S:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i8>{{.*}}
subroutine dshiftr1_test(a, b, s, c)
  integer(kind=1) :: a, b
  integer :: s
  integer(kind=1) :: c

  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[DS]]
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[DS]]
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[DS]]
  ! CHECK: %[[S_DECL:.*]]:2 = hlfir.declare %[[S]] dummy_scope %[[DS]]
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i8>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i8>
  ! CHECK: %[[S_VAL:.*]] = fir.load %[[S_DECL]]#0 : !fir.ref<i32>
  c = dshiftr(a, b, s)
  ! CHECK: %[[S_CONV:.*]] = fir.convert %[[S_VAL]] : (i32) -> i8
  ! CHECK: %[[C_BITS:.*]] = arith.constant 8 : i8
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C_BITS]], %[[S_CONV]] : i8
  ! CHECK: %[[C_BITS_L:.*]] = arith.constant 8 : i8
  ! CHECK: %[[C_0_L:.*]] = arith.constant 0 : i8
  ! CHECK: %[[UNDER_L:.*]] = arith.cmpi slt, %[[DIFF]], %[[C_0_L]] : i8
  ! CHECK: %[[OVER_L:.*]] = arith.cmpi sge, %[[DIFF]], %[[C_BITS_L]] : i8
  ! CHECK: %[[INV_L:.*]] = arith.ori %[[UNDER_L]], %[[OVER_L]] : i1
  ! CHECK: %[[SHL:.*]] = arith.shli %[[A_VAL]], %[[DIFF]] : i8
  ! CHECK: %[[LFT:.*]] = arith.select %[[INV_L]], %[[C_0_L]], %[[SHL]] : i8
  ! CHECK: %[[C_BITS_R:.*]] = arith.constant 8 : i8
  ! CHECK: %[[C_0_R:.*]] = arith.constant 0 : i8
  ! CHECK: %[[UNDER_R:.*]] = arith.cmpi slt, %[[S_CONV]], %[[C_0_R]] : i8
  ! CHECK: %[[OVER_R:.*]] = arith.cmpi sge, %[[S_CONV]], %[[C_BITS_R]] : i8
  ! CHECK: %[[INV_R:.*]] = arith.ori %[[UNDER_R]], %[[OVER_R]] : i1
  ! CHECK: %[[SHR:.*]] = arith.shrui %[[B_VAL]], %[[S_CONV]] : i8
  ! CHECK: %[[RGT:.*]] = arith.select %[[INV_R]], %[[C_0_R]], %[[SHR]] : i8
  ! CHECK: %[[SHIFT:.*]] = arith.ori %[[LFT]], %[[RGT]] : i8
  ! CHECK: hlfir.assign %[[SHIFT]] to %[[C_DECL]]#0 : i8, !fir.ref<i8>
end subroutine dshiftr1_test

! CHECK-LABEL: func.func @_QPdshiftr2_test(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16>{{.*}}, %[[B:.*]]: !fir.ref<i16>{{.*}}, %[[S:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i16>{{.*}}
subroutine dshiftr2_test(a, b, s, c)
  integer(kind=2) :: a, b
  integer :: s
  integer(kind=2) :: c

  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[DS]]
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[DS]]
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[DS]]
  ! CHECK: %[[S_DECL:.*]]:2 = hlfir.declare %[[S]] dummy_scope %[[DS]]
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i16>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i16>
  ! CHECK: %[[S_VAL:.*]] = fir.load %[[S_DECL]]#0 : !fir.ref<i32>
  c = dshiftr(a, b, s)
  ! CHECK: %[[S_CONV:.*]] = fir.convert %[[S_VAL]] : (i32) -> i16
  ! CHECK: %[[C_BITS:.*]] = arith.constant 16 : i16
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C_BITS]], %[[S_CONV]] : i16
  ! CHECK: %[[C_BITS_L:.*]] = arith.constant 16 : i16
  ! CHECK: %[[C_0_L:.*]] = arith.constant 0 : i16
  ! CHECK: %[[UNDER_L:.*]] = arith.cmpi slt, %[[DIFF]], %[[C_0_L]] : i16
  ! CHECK: %[[OVER_L:.*]] = arith.cmpi sge, %[[DIFF]], %[[C_BITS_L]] : i16
  ! CHECK: %[[INV_L:.*]] = arith.ori %[[UNDER_L]], %[[OVER_L]] : i1
  ! CHECK: %[[SHL:.*]] = arith.shli %[[A_VAL]], %[[DIFF]] : i16
  ! CHECK: %[[LFT:.*]] = arith.select %[[INV_L]], %[[C_0_L]], %[[SHL]] : i16
  ! CHECK: %[[C_BITS_R:.*]] = arith.constant 16 : i16
  ! CHECK: %[[C_0_R:.*]] = arith.constant 0 : i16
  ! CHECK: %[[UNDER_R:.*]] = arith.cmpi slt, %[[S_CONV]], %[[C_0_R]] : i16
  ! CHECK: %[[OVER_R:.*]] = arith.cmpi sge, %[[S_CONV]], %[[C_BITS_R]] : i16
  ! CHECK: %[[INV_R:.*]] = arith.ori %[[UNDER_R]], %[[OVER_R]] : i1
  ! CHECK: %[[SHR:.*]] = arith.shrui %[[B_VAL]], %[[S_CONV]] : i16
  ! CHECK: %[[RGT:.*]] = arith.select %[[INV_R]], %[[C_0_R]], %[[SHR]] : i16
  ! CHECK: %[[SHIFT:.*]] = arith.ori %[[LFT]], %[[RGT]] : i16
  ! CHECK: hlfir.assign %[[SHIFT]] to %[[C_DECL]]#0 : i16, !fir.ref<i16>
end subroutine dshiftr2_test

! CHECK-LABEL: func.func @_QPdshiftr4_test(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[S:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i32>{{.*}}
subroutine dshiftr4_test(a, b, s, c)
  integer(kind=4) :: a, b
  integer :: s
  integer(kind=4) :: c

  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[DS]]
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[DS]]
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[DS]]
  ! CHECK: %[[S_DECL:.*]]:2 = hlfir.declare %[[S]] dummy_scope %[[DS]]
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[S_VAL:.*]] = fir.load %[[S_DECL]]#0 : !fir.ref<i32>
  c = dshiftr(a, b, s)
  ! CHECK: %[[C_BITS:.*]] = arith.constant 32 : i32
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C_BITS]], %[[S_VAL]] : i32
  ! CHECK: %[[C_BITS_L:.*]] = arith.constant 32 : i32
  ! CHECK: %[[C_0_L:.*]] = arith.constant 0 : i32
  ! CHECK: %[[UNDER_L:.*]] = arith.cmpi slt, %[[DIFF]], %[[C_0_L]] : i32
  ! CHECK: %[[OVER_L:.*]] = arith.cmpi sge, %[[DIFF]], %[[C_BITS_L]] : i32
  ! CHECK: %[[INV_L:.*]] = arith.ori %[[UNDER_L]], %[[OVER_L]] : i1
  ! CHECK: %[[SHL:.*]] = arith.shli %[[A_VAL]], %[[DIFF]] : i32
  ! CHECK: %[[LFT:.*]] = arith.select %[[INV_L]], %[[C_0_L]], %[[SHL]] : i32
  ! CHECK: %[[C_BITS_R:.*]] = arith.constant 32 : i32
  ! CHECK: %[[C_0_R:.*]] = arith.constant 0 : i32
  ! CHECK: %[[UNDER_R:.*]] = arith.cmpi slt, %[[S_VAL]], %[[C_0_R]] : i32
  ! CHECK: %[[OVER_R:.*]] = arith.cmpi sge, %[[S_VAL]], %[[C_BITS_R]] : i32
  ! CHECK: %[[INV_R:.*]] = arith.ori %[[UNDER_R]], %[[OVER_R]] : i1
  ! CHECK: %[[SHR:.*]] = arith.shrui %[[B_VAL]], %[[S_VAL]] : i32
  ! CHECK: %[[RGT:.*]] = arith.select %[[INV_R]], %[[C_0_R]], %[[SHR]] : i32
  ! CHECK: %[[SHIFT:.*]] = arith.ori %[[LFT]], %[[RGT]] : i32
  ! CHECK: hlfir.assign %[[SHIFT]] to %[[C_DECL]]#0 : i32, !fir.ref<i32>
end subroutine dshiftr4_test

! CHECK-LABEL: func.func @_QPdshiftr8_test(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i64>{{.*}}, %[[B:.*]]: !fir.ref<i64>{{.*}}, %[[S:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i64>{{.*}}
subroutine dshiftr8_test(a, b, s, c)
  integer(kind=8) :: a, b
  integer :: s
  integer(kind=8) :: c

  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[DS]]
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[DS]]
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[DS]]
  ! CHECK: %[[S_DECL:.*]]:2 = hlfir.declare %[[S]] dummy_scope %[[DS]]
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i64>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i64>
  ! CHECK: %[[S_VAL:.*]] = fir.load %[[S_DECL]]#0 : !fir.ref<i32>
  c = dshiftr(a, b, s)
  ! CHECK: %[[S_CONV:.*]] = fir.convert %[[S_VAL]] : (i32) -> i64
  ! CHECK: %[[C_BITS:.*]] = arith.constant 64 : i64
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C_BITS]], %[[S_CONV]] : i64
  ! CHECK: %[[C_BITS_L:.*]] = arith.constant 64 : i64
  ! CHECK: %[[C_0_L:.*]] = arith.constant 0 : i64
  ! CHECK: %[[UNDER_L:.*]] = arith.cmpi slt, %[[DIFF]], %[[C_0_L]] : i64
  ! CHECK: %[[OVER_L:.*]] = arith.cmpi sge, %[[DIFF]], %[[C_BITS_L]] : i64
  ! CHECK: %[[INV_L:.*]] = arith.ori %[[UNDER_L]], %[[OVER_L]] : i1
  ! CHECK: %[[SHL:.*]] = arith.shli %[[A_VAL]], %[[DIFF]] : i64
  ! CHECK: %[[LFT:.*]] = arith.select %[[INV_L]], %[[C_0_L]], %[[SHL]] : i64
  ! CHECK: %[[C_BITS_R:.*]] = arith.constant 64 : i64
  ! CHECK: %[[C_0_R:.*]] = arith.constant 0 : i64
  ! CHECK: %[[UNDER_R:.*]] = arith.cmpi slt, %[[S_CONV]], %[[C_0_R]] : i64
  ! CHECK: %[[OVER_R:.*]] = arith.cmpi sge, %[[S_CONV]], %[[C_BITS_R]] : i64
  ! CHECK: %[[INV_R:.*]] = arith.ori %[[UNDER_R]], %[[OVER_R]] : i1
  ! CHECK: %[[SHR:.*]] = arith.shrui %[[B_VAL]], %[[S_CONV]] : i64
  ! CHECK: %[[RGT:.*]] = arith.select %[[INV_R]], %[[C_0_R]], %[[SHR]] : i64
  ! CHECK: %[[SHIFT:.*]] = arith.ori %[[LFT]], %[[RGT]] : i64
  ! CHECK: hlfir.assign %[[SHIFT]] to %[[C_DECL]]#0 : i64, !fir.ref<i64>
end subroutine dshiftr8_test

! CHECK-LABEL: func.func @_QPdshiftr16_test(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i128>{{.*}}, %[[B:.*]]: !fir.ref<i128>{{.*}}, %[[S:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i128>{{.*}}
subroutine dshiftr16_test(a, b, s, c)
  integer(kind=16) :: a, b
  integer :: s
  integer(kind=16) :: c

  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[DS]]
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[DS]]
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[DS]]
  ! CHECK: %[[S_DECL:.*]]:2 = hlfir.declare %[[S]] dummy_scope %[[DS]]
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i128>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i128>
  ! CHECK: %[[S_VAL:.*]] = fir.load %[[S_DECL]]#0 : !fir.ref<i32>
  c = dshiftr(a, b, s)
  ! CHECK: %[[S_CONV:.*]] = fir.convert %[[S_VAL]] : (i32) -> i128
  ! CHECK: %[[C_BITS:.*]] = arith.constant 128 : i128
  ! CHECK: %[[DIFF:.*]] = arith.subi %[[C_BITS]], %[[S_CONV]] : i128
  ! CHECK: %[[C_BITS_L:.*]] = arith.constant 128 : i128
  ! CHECK: %[[C_0_L:.*]] = arith.constant 0 : i128
  ! CHECK: %[[UNDER_L:.*]] = arith.cmpi slt, %[[DIFF]], %[[C_0_L]] : i128
  ! CHECK: %[[OVER_L:.*]] = arith.cmpi sge, %[[DIFF]], %[[C_BITS_L]] : i128
  ! CHECK: %[[INV_L:.*]] = arith.ori %[[UNDER_L]], %[[OVER_L]] : i1
  ! CHECK: %[[SHL:.*]] = arith.shli %[[A_VAL]], %[[DIFF]] : i128
  ! CHECK: %[[LFT:.*]] = arith.select %[[INV_L]], %[[C_0_L]], %[[SHL]] : i128
  ! CHECK: %[[C_BITS_R:.*]] = arith.constant 128 : i128
  ! CHECK: %[[C_0_R:.*]] = arith.constant 0 : i128
  ! CHECK: %[[UNDER_R:.*]] = arith.cmpi slt, %[[S_CONV]], %[[C_0_R]] : i128
  ! CHECK: %[[OVER_R:.*]] = arith.cmpi sge, %[[S_CONV]], %[[C_BITS_R]] : i128
  ! CHECK: %[[INV_R:.*]] = arith.ori %[[UNDER_R]], %[[OVER_R]] : i1
  ! CHECK: %[[SHR:.*]] = arith.shrui %[[B_VAL]], %[[S_CONV]] : i128
  ! CHECK: %[[RGT:.*]] = arith.select %[[INV_R]], %[[C_0_R]], %[[SHR]] : i128
  ! CHECK: %[[SHIFT:.*]] = arith.ori %[[LFT]], %[[RGT]] : i128
  ! CHECK: hlfir.assign %[[SHIFT]] to %[[C_DECL]]#0 : i128, !fir.ref<i128>
end subroutine dshiftr16_test
