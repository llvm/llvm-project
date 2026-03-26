! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPblt_test(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32> {{.*}}, %[[B:.*]]: !fir.ref<i32> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test(a, b, c)
  integer :: a, b
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_testEa"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_testEb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 3 {uniq_name = "_QFblt_testEc"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i32>
  c = blt(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_VAL]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test

! CHECK-LABEL: func.func @_QPblt_test1(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i8> {{.*}}, %[[B:.*]]: !fir.ref<i8> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test1(a, b, c)
  integer(kind=1) :: a, b
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test1Ea"} : (!fir.ref<i8>, !fir.dscope) -> (!fir.ref<i8>, !fir.ref<i8>)
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test1Eb"} : (!fir.ref<i8>, !fir.dscope) -> (!fir.ref<i8>, !fir.ref<i8>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 3 {uniq_name = "_QFblt_test1Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i8>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i8>
  c = blt(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_VAL]], %[[B_VAL]] : i8
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test1

! CHECK-LABEL: func.func @_QPblt_test2(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16> {{.*}}, %[[B:.*]]: !fir.ref<i16> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test2(a, b, c)
  integer(kind=2) :: a, b
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test2Ea"} : (!fir.ref<i16>, !fir.dscope) -> (!fir.ref<i16>, !fir.ref<i16>)
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test2Eb"} : (!fir.ref<i16>, !fir.dscope) -> (!fir.ref<i16>, !fir.ref<i16>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 3 {uniq_name = "_QFblt_test2Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i16>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i16>
  c = blt(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_VAL]], %[[B_VAL]] : i16
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test2

! CHECK-LABEL: func.func @_QPblt_test3(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32> {{.*}}, %[[B:.*]]: !fir.ref<i32> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test3(a, b, c)
  integer(kind=4) :: a, b
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test3Ea"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test3Eb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 3 {uniq_name = "_QFblt_test3Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i32>
  c = blt(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_VAL]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test3

! CHECK-LABEL: func.func @_QPblt_test4(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i64> {{.*}}, %[[B:.*]]: !fir.ref<i64> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test4(a, b, c)
  integer(kind=8) :: a, b
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test4Ea"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test4Eb"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 3 {uniq_name = "_QFblt_test4Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i64>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i64>
  c = blt(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_VAL]], %[[B_VAL]] : i64
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test4

! CHECK-LABEL: func.func @_QPblt_test5(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i128> {{.*}}, %[[B:.*]]: !fir.ref<i128> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test5(a, b, c)
  integer(kind=16) :: a, b
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test5Ea"} : (!fir.ref<i128>, !fir.dscope) -> (!fir.ref<i128>, !fir.ref<i128>)
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test5Eb"} : (!fir.ref<i128>, !fir.dscope) -> (!fir.ref<i128>, !fir.ref<i128>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 3 {uniq_name = "_QFblt_test5Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i128>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i128>
  c = blt(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_VAL]], %[[B_VAL]] : i128
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test5

! CHECK-LABEL: func.func @_QPblt_test6(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16> {{.*}}, %[[B:.*]]: !fir.ref<i32> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test6(a, b, c)
  integer(kind=2) :: a
  integer(kind=4) :: b
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test6Ea"} : (!fir.ref<i16>, !fir.dscope) -> (!fir.ref<i16>, !fir.ref<i16>)
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test6Eb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 3 {uniq_name = "_QFblt_test6Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i16>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i32>
  c = blt(a, b)
  ! CHECK: %[[A_EXT:.*]] = arith.extui %[[A_VAL]] : i16 to i32
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_EXT]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test6

! CHECK-LABEL: func.func @_QPblt_test7(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32> {{.*}}, %[[B:.*]]: !fir.ref<i16> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test7(a, b, c)
  integer(kind=4) :: a
  integer(kind=2) :: b
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test7Ea"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test7Eb"} : (!fir.ref<i16>, !fir.dscope) -> (!fir.ref<i16>, !fir.ref<i16>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 3 {uniq_name = "_QFblt_test7Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i16>
  c = blt(a, b)
  ! CHECK: %[[B_EXT:.*]] = arith.extui %[[B_VAL]] : i16 to i32
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_VAL]], %[[B_EXT]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test7

! CHECK-LABEL: func.func @_QPblt_test8(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test8(a, c)
  integer(kind=2) :: a
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test8Ea"} : (!fir.ref<i16>, !fir.dscope) -> (!fir.ref<i16>, !fir.ref<i16>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test8Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[B_VAL:.*]] = arith.constant 42 : i32
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i16>
  c = blt(a, 42_4)
  ! CHECK: %[[A_EXT:.*]] = arith.extui %[[A_VAL]] : i16 to i32
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_EXT]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test8

! CHECK-LABEL: func.func @_QPblt_test9(
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32> {{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test9(a, c)
  integer(kind=4) :: a
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test9Ea"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFblt_test9Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[B_VAL_K2:.*]] = arith.constant 42 : i16
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
  c = blt(a, 42_2)
  ! CHECK: %[[B_VAL:.*]] = arith.extui %[[B_VAL_K2]] : i16 to i32
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi ult, %[[A_VAL]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[C_VAL]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test9

! CHECK-LABEL: func.func @_QPblt_test10(
! CHECK-SAME: %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test10(c)
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test10Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  c = blt(-1_2, -1_4)
  ! CHECK: %[[R:.*]] = arith.constant true
  ! CHECK: %[[V:.*]] = fir.convert %[[R]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[V]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test10

! CHECK-LABEL: func.func @_QPblt_test11(
! CHECK-SAME: %[[C:.*]]: !fir.ref<!fir.logical<4>> {{.*}}) {
subroutine blt_test11(c)
  logical :: c
  ! CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFblt_test11Ec"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  c = blt(-1_4, -1_2)
  ! CHECK: %[[R:.*]] = arith.constant false
  ! CHECK: %[[V:.*]] = fir.convert %[[R]] : (i1) -> !fir.logical<4>
  ! CHECK: hlfir.assign %[[V]] to %[[C_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
end subroutine blt_test11
