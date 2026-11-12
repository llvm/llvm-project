! Test lowering of extent and lower bound inquires that
! come in lowering as evaluate::DescriptorInquiry.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_assumed_shape(x, r)
  integer(8) :: r
  real :: x(:,:)
  r = size(x, dim=2, kind=8)
end subroutine
! CHECK-LABEL: func.func @_QPtest_assumed_shape(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_4]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
! CHECK:  hlfir.assign %[[VAL_6]] to %{{.*}}

subroutine test_explicit_shape(x, n, m, r)
  integer(8) :: n, m, r
  real :: x(n,m)
  r = size(x, dim=2, kind=8)
end subroutine
! CHECK-LABEL: func.func @_QPtest_explicit_shape(
! CHECK:  %[[VAL_17:.*]] = fir.shape %{{.*}}, %[[VAL_16:.*]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_18:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_17]])  {{.*}}Ex
! CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_16]] : (index) -> i64
! CHECK:  hlfir.assign %[[VAL_19]] to %{{.*}}

subroutine test_pointer(x, r)
  integer(8) :: r
  real :: x(:,:)
  r = size(x, dim=2, kind=8)
end subroutine
! CHECK-LABEL: func.func @_QPtest_pointer(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_4]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
! CHECK:  hlfir.assign %[[VAL_6]] to %{{.*}}

subroutine test_lbound_assumed_shape(x, l1, l2, r)
  integer(8) :: l1, l2, r
  real :: x(l1:,l2:)
  r = lbound(x, dim=2, kind=8)
end subroutine
! CHECK:  %[[VAL_11:.*]] = fir.shift %[[VAL_8:.*]], %[[VAL_10:.*]] : (index, index) -> !fir.shift<2>
! CHECK:  %[[VAL_12:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_11]])  {{.*}}Ex
! CHECK:  %[[VAL_13:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_12]]#1, %[[VAL_15]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_16]]#1, %[[VAL_14]] : index
! CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:  %[[VAL_19:.*]] = arith.select %[[VAL_17]], %[[VAL_18]], %[[VAL_10]] : index
! CHECK:  %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
! CHECK:  hlfir.assign %[[VAL_20]] to %{{.*}}

subroutine test_lbound_explicit_shape(x, n, m, l1, l2, r)
  integer(8) :: n, m, l1, l2, r
  real :: x(l1:n,l2:m)
  r = lbound(x, dim=2, kind=8)
end subroutine
! CHECK-LABEL: func.func @_QPtest_lbound_explicit_shape(
! CHECK:  %[[VAL_31:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %[[VAL_22:.*]], %[[VAL_30:.*]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:  %[[VAL_32:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_31]])  {{.*}}Ex
! CHECK:  %[[VAL_33:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_34:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_35:.*]] = arith.cmpi eq, %[[VAL_30]], %[[VAL_34]] : index
! CHECK:  %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (i64) -> index
! CHECK:  %[[VAL_37:.*]] = arith.select %[[VAL_35]], %[[VAL_36]], %[[VAL_22]] : index
! CHECK:  %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (index) -> i64
! CHECK:  hlfir.assign %[[VAL_38]] to %{{.*}}

subroutine test_lbound_pointer(x, r)
  integer(8) :: r
  real, pointer :: x(:,:)
  r = lbound(x, dim=2, kind=8)
end subroutine
! CHECK-LABEL: func.func @_QPtest_lbound_pointer(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_5]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (index) -> i64
! CHECK:  hlfir.assign %[[VAL_7]] to %{{.*}}
