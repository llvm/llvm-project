! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPpoppar1_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i8>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine poppar1_test(a, b)
  integer(1) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = poppar(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i8>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i8
! CHECK:  %[[CONV:.*]] = fir.convert %[[COUNT]] : (i8) -> i32
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[RESULT:.*]] = arith.andi %[[CONV]], %[[C1]] : i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine poppar1_test

! CHECK-LABEL: func.func @_QPpoppar2_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i16>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine poppar2_test(a, b)
  integer(2) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = poppar(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i16>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i16
! CHECK:  %[[CONV:.*]] = fir.convert %[[COUNT]] : (i16) -> i32
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[RESULT:.*]] = arith.andi %[[CONV]], %[[C1]] : i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine poppar2_test

! CHECK-LABEL: func.func @_QPpoppar4_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i32>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine poppar4_test(a, b)
  integer(4) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = poppar(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i32
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[RESULT:.*]] = arith.andi %[[COUNT]], %[[C1]] : i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine poppar4_test

! CHECK-LABEL: func.func @_QPpoppar8_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i64>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine poppar8_test(a, b)
  integer(8) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = poppar(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i64>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i64
! CHECK:  %[[CONV:.*]] = fir.convert %[[COUNT]] : (i64) -> i32
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[RESULT:.*]] = arith.andi %[[CONV]], %[[C1]] : i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine poppar8_test

! CHECK-LABEL: func.func @_QPpoppar16_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i128>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine poppar16_test(a, b)
  integer(16) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = poppar(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i128>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i128
! CHECK:  %[[CONV:.*]] = fir.convert %[[COUNT]] : (i128) -> i32
! CHECK:  %[[C1:.*]] = arith.constant 1 : i32
! CHECK:  %[[RESULT:.*]] = arith.andi %[[CONV]], %[[C1]] : i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine poppar16_test
