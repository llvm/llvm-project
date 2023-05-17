! This test focus on cmplx with Y argument that may turn out
! to be absent at runtime because it is an unallocated allocatable,
! a disassociated pointer, or an optional argument.
! CMPLX without such argument is re-written by the front-end as a
! complex constructor that is tested elsewhere.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPcmplx_test_scalar_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>
subroutine cmplx_test_scalar_ptr(x, y)
  real :: x
  real, pointer :: y
  print *, cmplx(x, y)
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ptr<f32>) -> i64
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:  %[[VAL_13:.*]] = fir.if %[[VAL_12]] -> (f32) {
! CHECK:    %[[VAL_14:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:    %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:    %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ptr<f32>
! CHECK:    fir.result %[[VAL_16]] : f32
! CHECK:  } else {
! CHECK:    %[[VAL_17:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:    fir.result %[[VAL_17]] : f32
! CHECK:  }
! CHECK:  %[[VAL_18:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_19:.*]] = fir.insert_value %[[VAL_18]], %[[VAL_7]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:   fir.insert_value %[[VAL_19]], %[[VAL_21:.*]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
end subroutine

! CHECK-LABEL: func @_QPcmplx_test_scalar_optional(
! CHECK-SAME:  %[[VAL_0:[^:]*]]: !fir.ref<f32>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<f32>
subroutine cmplx_test_scalar_optional(x, y)
  real :: x
  real, optional :: y
  print *, cmplx(x, y)
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = fir.is_present %[[VAL_1]] : (!fir.ref<f32>) -> i1
! CHECK:  %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (f32) {
! CHECK:    %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<f32>
! CHECK:    fir.result %[[VAL_10]] : f32
! CHECK:  } else {
! CHECK:    %[[VAL_11:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:    fir.result %[[VAL_11]] : f32
! CHECK:  }
! CHECK:  %[[VAL_12:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_7]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  fir.insert_value %[[VAL_13]], %[[VAL_15:.*]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
end subroutine

! CHECK-LABEL: func @_QPcmplx_test_scalar_alloc_optional(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<i64>>>
subroutine cmplx_test_scalar_alloc_optional(x, y)
  real :: x
  integer(8), allocatable, optional :: y
  print *, cmplx(x, y)
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.heap<i64>) -> i64
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:  %[[VAL_13:.*]] = fir.if %[[VAL_12]] -> (i64) {
! CHECK:    %[[VAL_14:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:    %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:    %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.heap<i64>
! CHECK:    fir.result %[[VAL_16]] : i64
! CHECK:  } else {
! CHECK:    %[[VAL_17:.*]] = arith.constant 0 : i64
! CHECK:    fir.result %[[VAL_17]] : i64
! CHECK:  }
! CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_19:.*]] : (i64) -> f32
! CHECK:  %[[VAL_20:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_7]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  fir.insert_value %[[VAL_21]], %[[VAL_18]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
end subroutine

! CHECK-LABEL: func @_QPcmplx_test_pointer_result(
! CHECK-SAME:  %[[VAL_0:[^:]*]]: !fir.ref<f32>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<f32>
subroutine cmplx_test_pointer_result(x, y)
  real :: x
  interface
    function return_pointer()
      real, pointer :: return_pointer
    end function
  end interface
  print *, cmplx(x, return_pointer())
! CHECK:  %[[VAL_9:.*]] = fir.call @_QPreturn_pointer() {{.*}}: () -> !fir.box<!fir.ptr<f32>>
! CHECK:  fir.save_result %[[VAL_9]] to %[[VAL_2:.*]] : !fir.box<!fir.ptr<f32>>, !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[VAL_10:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ptr<f32>) -> i64
! CHECK:  %[[VAL_13:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:  %[[VAL_15:.*]] = fir.if %[[VAL_14]] -> (f32) {
! CHECK:    %[[VAL_16:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:    %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:    %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ptr<f32>
! CHECK:    fir.result %[[VAL_18]] : f32
! CHECK:  } else {
! CHECK:    %[[VAL_19:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:    fir.result %[[VAL_19]] : f32
! CHECK:  }
! CHECK:  %[[VAL_20:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_8]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  fir.insert_value %[[VAL_21]], %[[VAL_23:.*]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
end subroutine

! CHECK-LABEL: func @_QPcmplx_array(
! CHECK-SAME:  %[[VAL_0:[^:]*]]: !fir.box<!fir.array<?xf32>>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>
subroutine cmplx_array(x, y)
  ! Important, note that the shape is taken from `x` and not `y` that
  ! may be absent.
  real :: x(:)
  real, optional :: y(:)
  print *, cmplx(x, y)
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_7]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_9:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:  %[[VAL_10:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:  %[[VAL_11:.*]] = fir.zero_bits !fir.ref<!fir.array<?xf32>>
! CHECK:  %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_14:.*]] = fir.embox %[[VAL_11]](%[[VAL_13]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_15:.*]] = arith.select %[[VAL_10]], %[[VAL_1]], %[[VAL_14]] : !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_16:.*]] = fir.array_load %[[VAL_15]] {fir.optional} : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:  %[[VAL_17:.*]] = fir.allocmem !fir.array<?x!fir.complex<4>>, %[[VAL_8]]#1 {uniq_name = ".array.expr"}
! CHECK:  %[[VAL_18:.*]] = fir.shape %[[VAL_8]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_19:.*]] = fir.array_load %[[VAL_17]](%[[VAL_18]]) : (!fir.heap<!fir.array<?x!fir.complex<4>>>, !fir.shape<1>) -> !fir.array<?x!fir.complex<4>>
! CHECK:  %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_22:.*]] = arith.subi %[[VAL_8]]#1, %[[VAL_20]] : index
! CHECK:  %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_20]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_19]]) -> (!fir.array<?x!fir.complex<4>>) {
  ! CHECK:  %[[VAL_26:.*]] = fir.array_fetch %[[VAL_9]], %[[VAL_24]] : (!fir.array<?xf32>, index) -> f32
  ! CHECK:  %[[VAL_27:.*]] = fir.if %[[VAL_10]] -> (f32) {
    ! CHECK:  %[[VAL_28:.*]] = fir.array_fetch %[[VAL_16]], %[[VAL_24]] : (!fir.array<?xf32>, index) -> f32
    ! CHECK:  fir.result %[[VAL_28]] : f32
  ! CHECK:  } else {
    ! CHECK:  %[[VAL_29:.*]] = arith.constant 0.000000e+00 : f32
    ! CHECK:  fir.result %[[VAL_29]] : f32
  ! CHECK:  }
  ! CHECK:  %[[VAL_30:.*]] = fir.undefined !fir.complex<4>
  ! CHECK:  %[[VAL_31:.*]] = fir.insert_value %[[VAL_30]], %[[VAL_26]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
  ! CHECK:  %[[VAL_32:.*]] = fir.insert_value %[[VAL_31]], %[[VAL_33:.*]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
  ! CHECK:  %[[VAL_34:.*]] = fir.array_update %[[VAL_25]], %[[VAL_32]], %[[VAL_24]] : (!fir.array<?x!fir.complex<4>>, !fir.complex<4>, index) -> !fir.array<?x!fir.complex<4>>
  ! CHECK:  fir.result %[[VAL_34]] : !fir.array<?x!fir.complex<4>>
! CHECK:  }
! CHECK:  fir.array_merge_store
end subroutine
