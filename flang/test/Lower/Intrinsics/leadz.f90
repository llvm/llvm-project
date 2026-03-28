! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPleadz1_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i8>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine leadz1_test(a, b)
  integer(1) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = leadz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i8>
! CHECK:  %[[COUNT:.*]] = math.ctlz %[[AVAL]] : i8
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i8) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine leadz1_test

! CHECK-LABEL: func.func @_QPleadz2_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i16>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine leadz2_test(a, b)
  integer(2) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = leadz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i16>
! CHECK:  %[[COUNT:.*]] = math.ctlz %[[AVAL]] : i16
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i16) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine leadz2_test

! CHECK-LABEL: func.func @_QPleadz4_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i32>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine leadz4_test(a, b)
  integer(4) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = leadz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:  %[[RESULT:.*]] = math.ctlz %[[AVAL]] : i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine leadz4_test

! CHECK-LABEL: func.func @_QPleadz8_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i64>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine leadz8_test(a, b)
  integer(8) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = leadz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i64>
! CHECK:  %[[COUNT:.*]] = math.ctlz %[[AVAL]] : i64
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i64) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine leadz8_test

! CHECK-LABEL: func.func @_QPleadz16_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i128>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine leadz16_test(a, b)
  integer(16) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = leadz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i128>
! CHECK:  %[[COUNT:.*]] = math.ctlz %[[AVAL]] : i128
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i128) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine leadz16_test
