! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: bge_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test(a, b, c)
  integer :: a, b
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = bge(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_VAL]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test

! CHECK-LABEL: bge_test1
! CHECK-SAME: %[[A:.*]]: !fir.ref<i8>{{.*}}, %[[B:.*]]: !fir.ref<i8>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test1(a, b, c)
  integer(kind=1) :: a, b
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i8>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i8>
  c = bge(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_VAL]], %[[B_VAL]] : i8
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test1

! CHECK-LABEL: bge_test2
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16>{{.*}}, %[[B:.*]]: !fir.ref<i16>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test2(a, b, c)
  integer(kind=2) :: a, b
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i16>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i16>
  c = bge(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_VAL]], %[[B_VAL]] : i16
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test2

! CHECK-LABEL: bge_test3
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test3(a, b, c)
  integer(kind=4) :: a, b
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = bge(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_VAL]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test3

! CHECK-LABEL: bge_test4
! CHECK-SAME: %[[A:.*]]: !fir.ref<i64>{{.*}}, %[[B:.*]]: !fir.ref<i64>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test4(a, b, c)
  integer(kind=8) :: a, b
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i64>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i64>
  c = bge(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_VAL]], %[[B_VAL]] : i64
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test4

! CHECK-LABEL: bge_test5
! CHECK-SAME: %[[A:.*]]: !fir.ref<i128>{{.*}}, %[[B:.*]]: !fir.ref<i128>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test5(a, b, c)
  integer(kind=16) :: a, b
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i128>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i128>
  c = bge(a, b)
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_VAL]], %[[B_VAL]] : i128
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test5

! CHECK-LABEL: bge_test6
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test6(a, b, c)
  integer(kind=2) :: a
  integer(kind=4) :: b
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i16>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = bge(a, b)
  ! CHECK: %[[A_EXT:.*]] = arith.extui %[[A_VAL]] : i16 to i32
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_EXT]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test6

! CHECK-LABEL: bge_test7
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i16>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test7(a, b, c)
  integer(kind=4) :: a
  integer(kind=2) :: b
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i16>
  c = bge(a, b)
  ! CHECK: %[[B_EXT:.*]] = arith.extui %[[B_VAL]] : i16 to i32
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_VAL]], %[[B_EXT]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test7

! CHECK-LABEL: bge_test8
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test8(a, c)
  integer(kind=2) :: a
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i16>
  ! CHECK: %[[B_VAL:.*]] = arith.constant 42 : i32
  c = bge(a, 42_4)
  ! CHECK: %[[A_EXT:.*]] = arith.extui %[[A_VAL]] : i16 to i32
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_EXT]], %[[B_VAL]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test8

! CHECK-LABEL: bge_test9
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test9(a, c)
  integer(kind=4) :: a
  logical :: c
  ! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
  ! CHECK: %[[B_VAL:.*]] = arith.constant 42 : i16
  c = bge(a, 42_2)
  ! CHECK: %[[B_EXT:.*]] = arith.extui %[[B_VAL]] : i16 to i32
  ! CHECK: %[[C_CMP:.*]] = arith.cmpi uge, %[[A_VAL]], %[[B_EXT]] : i32
  ! CHECK: %[[C_VAL:.*]] = fir.convert %[[C_CMP]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test9

! CHECK-LABEL: bge_test10
! CHECK-SAME: %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test10(c)
  logical :: c
  c = bge(-1_2, -1_4)
  ! CHECK: %[[R:.*]] = arith.constant false
  ! CHECK: %[[V:.*]] = fir.convert %[[R]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[V]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test10

! CHECK-LABEL: bge_test11
! CHECK-SAME: %[[C:.*]]: !fir.ref<!fir.logical<4>>{{.*}}
subroutine bge_test11(c)
  logical :: c
  c = bge(-1_4, -1_2)
  ! CHECK: %[[R:.*]] = arith.constant true
  ! CHECK: %[[V:.*]] = fir.convert %[[R]] : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[V]] to %[[C]] : !fir.ref<!fir.logical<4>>
end subroutine bge_test11
