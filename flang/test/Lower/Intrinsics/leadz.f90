! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: leadz1_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i8>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine leadz1_test(a, b)
  integer(1) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i8>
  b = leadz(a)
  ! CHECK:  %[[COUNT:.*]] = math.ctlz %[[AVAL]] : i8
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i8) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine leadz1_test

! CHECK-LABEL: leadz2_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i16>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine leadz2_test(a, b)
  integer(2) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i16>
  b = leadz(a)
  ! CHECK:  %[[COUNT:.*]] = math.ctlz %[[AVAL]] : i16
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i16) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine leadz2_test

! CHECK-LABEL: leadz4_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i32>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine leadz4_test(a, b)
  integer(4) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i32>
  b = leadz(a)
  ! CHECK:  %[[RESULT:.*]] = math.ctlz %[[AVAL]] : i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine leadz4_test

! CHECK-LABEL: leadz8_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i64>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine leadz8_test(a, b)
  integer(8) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i64>
  b = leadz(a)
  ! CHECK:  %[[COUNT:.*]] = math.ctlz %[[AVAL]] : i64
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i64) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine leadz8_test

! CHECK-LABEL: leadz16_test
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<i128>{{.*}}, %[[BREF:.*]]: !fir.ref<i32>{{.*}}
subroutine leadz16_test(a, b)
  integer(16) :: a
  integer :: b

  ! CHECK:  %[[AVAL:.*]] = fir.load %[[AREF]] : !fir.ref<i128>
  b = leadz(a)
  ! CHECK:  %[[COUNT:.*]] = math.ctlz %[[AVAL]] : i128
  ! CHECK:  %[[RESULT:.*]] = fir.convert %[[COUNT]] : (i128) -> i32
  ! CHECK:  fir.store %[[RESULT]] to %[[BREF]] : !fir.ref<i32>
end subroutine leadz16_test
