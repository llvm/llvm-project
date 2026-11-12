! Test calls with POINTER dummy arguments on the callee side.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPchar_assumed(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{.*}}
subroutine char_assumed(a)
  integer :: n
  character(len=*), pointer :: a
  ! CHECK: %[[argLoad:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK: %[[argLen:.*]] = fir.box_elesize %[[argLoad]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index

  n = len(a)
  ! CHECK: %[[argLenCast:.*]] = fir.convert %[[argLen]] : (index) -> i32
  ! CHECK: fir.store %[[argLenCast]] to %{{.*}} : !fir.ref<i32>
end subroutine

! CHECK-LABEL: func @_QPchar_assumed_optional(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{.*}}
subroutine char_assumed_optional(a)
  integer :: n
  character(len=*), pointer, optional :: a
  ! CHECK: %[[argPresent:.*]] = fir.is_present %[[arg0]] : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> i1
  ! CHECK: %[[argLen:.*]] = fir.if %[[argPresent]] -> (index) {
  ! CHECK:   %[[argLoad:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK:   %[[argLoadLen:.*]] = fir.box_elesize %[[argLoad]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
  ! CHECK:   fir.result %[[argLoadLen]] : index
  ! CHECK: } else {
  ! CHECK:   %[[undef:.*]] = fir.undefined index
  ! CHECK:   fir.result %[[undef]] : index
  ! CHECK: }

  if (present(a)) then
    n = len(a)
    ! CHECK:   %[[argLenCast:.*]] = fir.convert %[[argLen]] : (index) -> i32
    ! CHECK:   fir.store %[[argLenCast]] to %{{.*}} : !fir.ref<i32>
  endif
end subroutine
