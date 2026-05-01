! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtrailz1_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i8>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine trailz1_test(a, b)
  integer(1) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = trailz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i8>
! CHECK:  %[[COUNT:.*]] = math.cttz %[[AVAL]] : i8
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i8) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine trailz1_test

! CHECK-LABEL: func.func @_QPtrailz2_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i16>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine trailz2_test(a, b)
  integer(2) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = trailz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i16>
! CHECK:  %[[COUNT:.*]] = math.cttz %[[AVAL]] : i16
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i16) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine trailz2_test

! CHECK-LABEL: func.func @_QPtrailz4_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i32>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine trailz4_test(a, b)
  integer(4) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = trailz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:  %[[RESULT:.*]] = math.cttz %[[AVAL]] : i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine trailz4_test

! CHECK-LABEL: func.func @_QPtrailz8_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i64>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine trailz8_test(a, b)
  integer(8) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = trailz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i64>
! CHECK:  %[[COUNT:.*]] = math.cttz %[[AVAL]] : i64
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i64) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine trailz8_test

! CHECK-LABEL: func.func @_QPtrailz16_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i128>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine trailz16_test(a, b)
  integer(16) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = trailz(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i128>
! CHECK:  %[[COUNT:.*]] = math.cttz %[[AVAL]] : i128
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i128) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine trailz16_test
