! This test focus on cmplx with Y argument that may turn out
! to be absent at runtime because it is an unallocated allocatable,
! a disassociated pointer, or an optional argument.
! CMPLX without such argument is re-written by the front-end as a
! complex constructor that is tested elsewhere.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcmplx_test_scalar_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "y"})
subroutine cmplx_test_scalar_ptr(x, y)
  real :: x
  real, pointer :: y
  print *, cmplx(x, y)
! CHECK-DAG:  %[[X:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK-DAG:  %[[Y:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:  %[[VAL_8:.*]] = fir.load %[[Y]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ptr<f32>) -> i64
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:  %[[VAL_7:.*]] = fir.load %[[X]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_13:.*]] = fir.if %[[VAL_12]] -> (f32) {
! CHECK:    %[[VAL_14:.*]] = fir.load %[[Y]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:    %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:    %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ptr<f32>
! CHECK:    fir.result %[[VAL_16]] : f32
! CHECK:  } else {
! CHECK:    %[[VAL_17:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:    fir.result %[[VAL_17]] : f32
! CHECK:  }
! CHECK:  %[[VAL_18:.*]] = fir.undefined complex<f32>
! CHECK:  %[[VAL_19:.*]] = fir.insert_value %[[VAL_18]], %[[VAL_7]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:  %[[VAL_20:.*]] = fir.insert_value %[[VAL_19]], %[[VAL_13]], [1 : index] : (complex<f32>, f32) -> complex<f32>
end subroutine

! CHECK-LABEL: func.func @_QPcmplx_test_scalar_optional(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}, %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "y", fir.optional})
subroutine cmplx_test_scalar_optional(x, y)
  real :: x
  real, optional :: y
  print *, cmplx(x, y)
! CHECK-DAG:  %[[X:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK-DAG:  %[[Y:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:  %[[VAL_8:.*]] = fir.is_present %[[Y]]#0 : (!fir.ref<f32>) -> i1
! CHECK:  %[[VAL_7:.*]] = fir.load %[[X]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (f32) {
! CHECK:    %[[VAL_10:.*]] = fir.load %[[Y]]#0 : !fir.ref<f32>
! CHECK:    fir.result %[[VAL_10]] : f32
! CHECK:  } else {
! CHECK:    %[[VAL_11:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:    fir.result %[[VAL_11]] : f32
! CHECK:  }
! CHECK:  %[[VAL_12:.*]] = fir.undefined complex<f32>
! CHECK:  %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_7]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:  %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_9]], [1 : index] : (complex<f32>, f32) -> complex<f32>
end subroutine

! CHECK-LABEL: func.func @_QPcmplx_test_scalar_alloc_optional(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<i64>>> {fir.bindc_name = "y", fir.optional})
subroutine cmplx_test_scalar_alloc_optional(x, y)
  real :: x
  integer(8), allocatable, optional :: y
  print *, cmplx(x, y)
! CHECK-DAG:  %[[X:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK-DAG:  %[[Y:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:  %[[VAL_8:.*]] = fir.load %[[Y]]#0 : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.heap<i64>) -> i64
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:  %[[VAL_7:.*]] = fir.load %[[X]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_13:.*]] = fir.if %[[VAL_12]] -> (i64) {
! CHECK:    %[[VAL_14:.*]] = fir.load %[[Y]]#0 : !fir.ref<!fir.box<!fir.heap<i64>>>
! CHECK:    %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
! CHECK:    %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.heap<i64>
! CHECK:    fir.result %[[VAL_16]] : i64
! CHECK:  } else {
! CHECK:    %[[VAL_17:.*]] = arith.constant 0 : i64
! CHECK:    fir.result %[[VAL_17]] : i64
! CHECK:  }
! CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_13]] : (i64) -> f32
! CHECK:  %[[VAL_20:.*]] = fir.undefined complex<f32>
! CHECK:  %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_7]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:  %[[VAL_22:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_18]], [1 : index] : (complex<f32>, f32) -> complex<f32>
end subroutine

! CHECK-LABEL: func.func @_QPcmplx_test_pointer_result(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}, %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "y"})
subroutine cmplx_test_pointer_result(x, y)
  real :: x
  interface
    function return_pointer()
      real, pointer :: return_pointer
    end function
  end interface
  print *, cmplx(x, return_pointer())
! CHECK-DAG:  %[[X:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:  %[[VAL_9:.*]] = fir.call @_QPreturn_pointer()
! CHECK:  fir.save_result %[[VAL_9]] to %[[RESULT_ALLOCA:.*]] : !fir.box<!fir.ptr<f32>>, !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[RESULT_DECL:.*]]:2 = hlfir.declare %[[RESULT_ALLOCA]]
! CHECK:  %[[VAL_10:.*]] = fir.load %[[RESULT_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ptr<f32>) -> i64
! CHECK:  %[[VAL_13:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:  %[[VAL_8:.*]] = fir.load %[[X]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_15:.*]] = fir.if %[[VAL_14]] -> (f32) {
! CHECK:    %[[VAL_16:.*]] = fir.load %[[RESULT_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:    %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:    %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ptr<f32>
! CHECK:    fir.result %[[VAL_18]] : f32
! CHECK:  } else {
! CHECK:    %[[VAL_19:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:    fir.result %[[VAL_19]] : f32
! CHECK:  }
! CHECK:  %[[VAL_20:.*]] = fir.undefined complex<f32>
! CHECK:  %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_8]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:  %[[VAL_22:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_15]], [1 : index] : (complex<f32>, f32) -> complex<f32>
end subroutine

! CHECK-LABEL: func.func @_QPcmplx_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y", fir.optional})
subroutine cmplx_array(x, y)
  ! Important, note that the shape is taken from `x` and not `y` that
  ! may be absent.
  real :: x(:)
  real, optional :: y(:)
  print *, cmplx(x, y)
! CHECK-DAG:  %[[X:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK-DAG:  %[[Y:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[Y]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:  %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xcomplex<f32>> {
! CHECK:  ^bb0(%[[VAL_24:.*]]: index):
! CHECK:    %[[VAL_26_REF:.*]] = hlfir.designate %[[X]]#0 (%[[VAL_24]])
! CHECK:    %[[VAL_26:.*]] = fir.load %[[VAL_26_REF]] : !fir.ref<f32>
! CHECK:    %[[VAL_27:.*]] = fir.if %[[IS_PRESENT]] -> (f32) {
! CHECK:      %[[VAL_28_REF:.*]] = hlfir.designate %[[Y]]#0 (%[[VAL_24]])
! CHECK:      %[[VAL_28:.*]] = fir.load %[[VAL_28_REF]] : !fir.ref<f32>
! CHECK:      fir.result %[[VAL_28]] : f32
! CHECK:    } else {
! CHECK:      %[[VAL_29:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:      fir.result %[[VAL_29]] : f32
! CHECK:    }
! CHECK:    %[[VAL_30:.*]] = fir.undefined complex<f32>
! CHECK:    %[[VAL_31:.*]] = fir.insert_value %[[VAL_30]], %[[VAL_26]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:    %[[VAL_32:.*]] = fir.insert_value %[[VAL_31]], %[[VAL_27]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:    hlfir.yield_element %[[VAL_32]] : complex<f32>
! CHECK:  }
end subroutine
