! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: iand_test
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i32>{{.*}}
subroutine iand_test(a, b, c)
  integer :: a, b, c
! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<i32>
end subroutine iand_test

! CHECK-LABEL: iand_test1
! CHECK-SAME: %[[A:.*]]: !fir.ref<i8>{{.*}}, %[[B:.*]]: !fir.ref<i8>{{.*}}, %[[C:.*]]: !fir.ref<i8>{{.*}}
subroutine iand_test1(a, b, c)
  integer(kind=1) :: a, b, c
! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i8>
! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i8>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i8
! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<i8>
end subroutine iand_test1

! CHECK-LABEL: iand_test2
! CHECK-SAME: %[[A:.*]]: !fir.ref<i16>{{.*}}, %[[B:.*]]: !fir.ref<i16>{{.*}}, %[[C:.*]]: !fir.ref<i16>{{.*}}
subroutine iand_test2(a, b, c)
  integer(kind=2) :: a, b, c
! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i16>
! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i16>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i16
! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<i16>
end subroutine iand_test2

! CHECK-LABEL: iand_test3
! CHECK-SAME: %[[A:.*]]: !fir.ref<i32>{{.*}}, %[[B:.*]]: !fir.ref<i32>{{.*}}, %[[C:.*]]: !fir.ref<i32>{{.*}}
subroutine iand_test3(a, b, c)
  integer(kind=4) :: a, b, c
! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<i32>
end subroutine iand_test3

! CHECK-LABEL: iand_test4
! CHECK-SAME: %[[A:.*]]: !fir.ref<i64>{{.*}}, %[[B:.*]]: !fir.ref<i64>{{.*}}, %[[C:.*]]: !fir.ref<i64>{{.*}}
subroutine iand_test4(a, b, c)
  integer(kind=8) :: a, b, c
! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i64>
! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i64>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i64
! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<i64>
end subroutine iand_test4

! CHECK-LABEL: iand_test5
! CHECK-SAME: %[[A:.*]]: !fir.ref<i128>{{.*}}, %[[B:.*]]: !fir.ref<i128>{{.*}}, %[[C:.*]]: !fir.ref<i128>{{.*}}
subroutine iand_test5(a, b, c)
  integer(kind=16) :: a, b, c
! CHECK: %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i128>
! CHECK: %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i128>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i128
! CHECK: fir.store %[[C_VAL]] to %[[C]] : !fir.ref<i128>
end subroutine iand_test5

! CHECK-LABEL: iand_test6
! CHECK-SAME: %[[S1:.*]]: !fir.ref<i32>{{.*}}, %[[S2:.*]]: !fir.ref<i32>{{.*}}
subroutine iand_test6(s1, s2)
  integer :: s1, s2
! CHECK-DAG: %[[S1_VAL:.*]] = fir.load %[[S1]] : !fir.ref<i32>
! CHECK-DAG: %[[S2_VAL:.*]] = fir.load %[[S2]] : !fir.ref<i32>
  stop iand(s1,s2)
! CHECK-DAG: %[[ANDI:.*]] = arith.andi %[[S1_VAL]], %[[S2_VAL]] : i32
! CHECK: fir.call @_FortranAStopStatement(%[[ANDI]], {{.*}}, {{.*}}) : (i32, i1, i1) -> none
! CHECK-NEXT: fir.unreachable
end subroutine iand_test6
