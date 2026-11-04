! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPiand_test(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[B_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[C_ARG:.*]]: !fir.ref<i32>{{.*}})
subroutine iand_test(a, b, c)
  integer :: a, b, c
! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-DAG: %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
! CHECK-DAG: %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK: hlfir.assign %[[C_VAL]] to %[[C]]#0 : i32, !fir.ref<i32>
end subroutine iand_test

! CHECK-LABEL: func.func @_QPiand_test1(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.ref<i8>{{.*}}, %[[B_ARG:.*]]: !fir.ref<i8>{{.*}}, %[[C_ARG:.*]]: !fir.ref<i8>{{.*}})
subroutine iand_test1(a, b, c)
  integer(kind=1) :: a, b, c
! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-DAG: %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
! CHECK-DAG: %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i8>
! CHECK-DAG: %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i8>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i8
! CHECK: hlfir.assign %[[C_VAL]] to %[[C]]#0 : i8, !fir.ref<i8>
end subroutine iand_test1

! CHECK-LABEL: func.func @_QPiand_test2(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.ref<i16>{{.*}}, %[[B_ARG:.*]]: !fir.ref<i16>{{.*}}, %[[C_ARG:.*]]: !fir.ref<i16>{{.*}})
subroutine iand_test2(a, b, c)
  integer(kind=2) :: a, b, c
! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-DAG: %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
! CHECK-DAG: %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i16>
! CHECK-DAG: %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i16>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i16
! CHECK: hlfir.assign %[[C_VAL]] to %[[C]]#0 : i16, !fir.ref<i16>
end subroutine iand_test2

! CHECK-LABEL: func.func @_QPiand_test3(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[B_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[C_ARG:.*]]: !fir.ref<i32>{{.*}})
subroutine iand_test3(a, b, c)
  integer(kind=4) :: a, b, c
! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-DAG: %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
! CHECK-DAG: %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK: hlfir.assign %[[C_VAL]] to %[[C]]#0 : i32, !fir.ref<i32>
end subroutine iand_test3

! CHECK-LABEL: func.func @_QPiand_test4(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.ref<i64>{{.*}}, %[[B_ARG:.*]]: !fir.ref<i64>{{.*}}, %[[C_ARG:.*]]: !fir.ref<i64>{{.*}})
subroutine iand_test4(a, b, c)
  integer(kind=8) :: a, b, c
! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-DAG: %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
! CHECK-DAG: %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i64>
! CHECK-DAG: %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i64>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i64
! CHECK: hlfir.assign %[[C_VAL]] to %[[C]]#0 : i64, !fir.ref<i64>
end subroutine iand_test4

! CHECK-LABEL: func.func @_QPiand_test5(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.ref<i128>{{.*}}, %[[B_ARG:.*]]: !fir.ref<i128>{{.*}}, %[[C_ARG:.*]]: !fir.ref<i128>{{.*}})
subroutine iand_test5(a, b, c)
  integer(kind=16) :: a, b, c
! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-DAG: %[[C:.*]]:2 = hlfir.declare %[[C_ARG]]
! CHECK-DAG: %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i128>
! CHECK-DAG: %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i128>
  c = iand(a, b)
! CHECK: %[[C_VAL:.*]] = arith.andi %[[A_VAL]], %[[B_VAL]] : i128
! CHECK: hlfir.assign %[[C_VAL]] to %[[C]]#0 : i128, !fir.ref<i128>
end subroutine iand_test5

! CHECK-LABEL: func.func @_QPiand_test6(
! CHECK-SAME: %[[S1_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[S2_ARG:.*]]: !fir.ref<i32>{{.*}})
subroutine iand_test6(s1, s2)
  integer :: s1, s2
! CHECK-DAG: %[[S1:.*]]:2 = hlfir.declare %[[S1_ARG]]
! CHECK-DAG: %[[S2:.*]]:2 = hlfir.declare %[[S2_ARG]]
! CHECK-DAG: %[[S1_VAL:.*]] = fir.load %[[S1]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[S2_VAL:.*]] = fir.load %[[S2]]#0 : !fir.ref<i32>
  stop iand(s1,s2)
! CHECK-DAG: %[[ANDI:.*]] = arith.andi %[[S1_VAL]], %[[S2_VAL]] : i32
! CHECK: fir.call @_FortranAStopStatement(%[[ANDI]], {{.*}}, {{.*}}) {{.*}}: (i32, i1, i1) -> ()
! CHECK-NEXT: fir.unreachable
end subroutine iand_test6
