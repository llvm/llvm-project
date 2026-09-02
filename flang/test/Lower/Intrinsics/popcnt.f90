! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPpopcnt1_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i8>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine popcnt1_test(a, b)
  integer(1) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = popcnt(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i8>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i8
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i8) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine popcnt1_test

! CHECK-LABEL: func.func @_QPpopcnt2_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i16>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine popcnt2_test(a, b)
  integer(2) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = popcnt(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i16>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i16
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i16) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine popcnt2_test

! CHECK-LABEL: func.func @_QPpopcnt4_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i32>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine popcnt4_test(a, b)
  integer(4) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = popcnt(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:  %[[RESULT:.*]] = math.ctpop %[[AVAL]] : i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine popcnt4_test

! CHECK-LABEL: func.func @_QPpopcnt8_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i64>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine popcnt8_test(a, b)
  integer(8) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = popcnt(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i64>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i64
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i64) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine popcnt8_test

! CHECK-LABEL: func.func @_QPpopcnt16_test(
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i128>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}})
subroutine popcnt16_test(a, b)
  integer(16) :: a
  integer :: b

! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare %[[AREF]]
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare %[[BREF]]
  b = popcnt(a)
! CHECK-DAG:  %[[AVAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i128>
! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i128
! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i128) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[B]]#0 : i32, !fir.ref<i32>
end subroutine popcnt16_test
