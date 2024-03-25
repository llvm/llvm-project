! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: popcnt1_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i8>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine popcnt1_test(a, b)
  integer(1) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i8>
  b = popcnt(a)
  ! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i8
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i8) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine popcnt1_test

! CHECK-LABEL: popcnt2_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i16>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine popcnt2_test(a, b)
  integer(2) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i16>
  b = popcnt(a)
  ! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i16
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i16) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine popcnt2_test

! CHECK-LABEL: popcnt4_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i32>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine popcnt4_test(a, b)
  integer(4) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i32>
  b = popcnt(a)
  ! CHECK:  %[[RESULT:.*]] = math.ctpop %[[AVAL]] : i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine popcnt4_test

! CHECK-LABEL: popcnt8_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i64>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine popcnt8_test(a, b)
  integer(8) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i64>
  b = popcnt(a)
  ! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i64
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i64) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine popcnt8_test

! CHECK-LABEL: popcnt16_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i128>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine popcnt16_test(a, b)
  integer(16) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i128>
  b = popcnt(a)
  ! CHECK:  %[[COUNT:.*]] = math.ctpop %[[AVAL]] : i128
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i128) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine popcnt16_test
