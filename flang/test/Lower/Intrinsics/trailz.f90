! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: trailz1_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i8>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine trailz1_test(a, b)
  integer(1) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i8>
  b = trailz(a)
  ! CHECK:  %[[COUNT:.*]] = math.cttz %[[AVAL]] : i8
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i8) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine trailz1_test

! CHECK-LABEL: trailz2_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i16>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine trailz2_test(a, b)
  integer(2) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i16>
  b = trailz(a)
  ! CHECK:  %[[COUNT:.*]] = math.cttz %[[AVAL]] : i16
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i16) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine trailz2_test

! CHECK-LABEL: trailz4_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i32>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine trailz4_test(a, b)
  integer(4) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i32>
  b = trailz(a)
  ! CHECK:  %[[RESULT:.*]] = math.cttz %[[AVAL]] : i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine trailz4_test

! CHECK-LABEL: trailz8_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i64>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine trailz8_test(a, b)
  integer(8) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i64>
  b = trailz(a)
  ! CHECK:  %[[COUNT:.*]] = math.cttz %[[AVAL]] : i64
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i64) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine trailz8_test

! CHECK-LABEL: trailz16_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i128>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine trailz16_test(a, b)
  integer(16) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i128>
  b = trailz(a)
  ! CHECK:  %[[COUNT:.*]] = math.cttz %[[AVAL]] : i128
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i128) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine trailz16_test
