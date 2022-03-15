! RUN: bbc -emit-fir %s -o - | FileCheck %s

module max_test
    contains
    ! CHECK-LABEL: func @_QMmax_testPdynamic_optional(
    ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
    ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"},
    ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c", fir.optional}) {
    subroutine dynamic_optional(a, b, c)
      integer :: a(:), b(:)
      integer, optional :: c(:)
    ! CHECK:  %[[VAL_10:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
    ! CHECK:  %[[VAL_11:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
    ! CHECK:  %[[VAL_12:.*]] = fir.is_present %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>) -> i1
    ! CHECK:  %[[VAL_17:.*]] = arith.select %[[VAL_12]], %[[VAL_2]], %{{.*}} : !fir.box<!fir.array<?xi32>>
    ! CHECK:  %[[VAL_18:.*]] = fir.array_load %[[VAL_17]] {fir.optional} : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
    ! CHECK:  fir.do_loop %[[VAL_26:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_27:.*]] = %{{.*}}) -> (!fir.array<?xi32>) {
    ! CHECK:    %[[VAL_28:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_26]] : (!fir.array<?xi32>, index) -> i32
    ! CHECK:    %[[VAL_29:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_26]] : (!fir.array<?xi32>, index) -> i32
    ! CHECK:    %[[VAL_30:.*]] = arith.cmpi sgt, %[[VAL_28]], %[[VAL_29]] : i32
    ! CHECK:    %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_28]], %[[VAL_29]] : i32
    ! CHECK:    %[[VAL_32:.*]] = fir.if %[[VAL_12]] -> (i32) {
    ! CHECK:      %[[VAL_33:.*]] = fir.array_fetch %[[VAL_18]], %[[VAL_26]] : (!fir.array<?xi32>, index) -> i32
    ! CHECK:      %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_33]] : i32
    ! CHECK:      %[[VAL_35:.*]] = arith.select %[[VAL_34]], %[[VAL_31]], %[[VAL_33]] : i32
    ! CHECK:      fir.result %[[VAL_35]] : i32
    ! CHECK:    } else {
    ! CHECK:      fir.result %[[VAL_31]] : i32
    ! CHECK:    }
    ! CHECK:    %[[VAL_36:.*]] = fir.array_update %[[VAL_27]], %[[VAL_32]], %[[VAL_26]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
    ! CHECK:    fir.result %[[VAL_36]] : !fir.array<?xi32>
    ! CHECK:  }
      print *, max(a, b, c)
    end subroutine 
    
    ! CHECK-LABEL: func @_QMmax_testPdynamic_optional_array_expr_scalar_optional(
    ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
    ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"},
    ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional}) {
    subroutine dynamic_optional_array_expr_scalar_optional(a, b, c)
      integer :: a(:), b(:)
      integer, optional :: c
      print *, max(a, b, c)
    ! CHECK:  %[[VAL_10:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
    ! CHECK:  %[[VAL_11:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
    ! CHECK:  %[[VAL_12:.*]] = fir.is_present %[[VAL_2]] : (!fir.ref<i32>) -> i1
    ! CHECK:  fir.do_loop %[[VAL_20:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_21:.*]] = %{{.*}}) -> (!fir.array<?xi32>) {
    ! CHECK:    %[[VAL_22:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_20]] : (!fir.array<?xi32>, index) -> i32
    ! CHECK:    %[[VAL_23:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_20]] : (!fir.array<?xi32>, index) -> i32
    ! CHECK:    %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_22]], %[[VAL_23]] : i32
    ! CHECK:    %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_22]], %[[VAL_23]] : i32
    ! CHECK:    %[[VAL_26:.*]] = fir.if %[[VAL_12]] -> (i32) {
    ! CHECK:      %[[VAL_27:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
    ! CHECK:      %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_25]], %[[VAL_27]] : i32
    ! CHECK:      %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_25]], %[[VAL_27]] : i32
    ! CHECK:      fir.result %[[VAL_29]] : i32
    ! CHECK:    } else {
    ! CHECK:      fir.result %[[VAL_25]] : i32
    ! CHECK:    }
    ! CHECK:    %[[VAL_30:.*]] = fir.array_update %[[VAL_21]], %[[VAL_26]], %[[VAL_20]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
    ! CHECK:    fir.result %[[VAL_30]] : !fir.array<?xi32>
    ! CHECK:  }
    end subroutine 
    
    ! CHECK-LABEL: func @_QMmax_testPdynamic_optional_scalar(
    ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
    ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"},
    ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional}) {
    subroutine dynamic_optional_scalar(a, b, c)
      integer :: a, b
      integer, optional :: c
      print *, max(a, b, c)
    ! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
    ! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
    ! CHECK:  %[[VAL_10:.*]] = fir.is_present %[[VAL_2]] : (!fir.ref<i32>) -> i1
    ! CHECK:  %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : i32
    ! CHECK:  %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_8]], %[[VAL_9]] : i32
    ! CHECK:  %[[VAL_13:.*]] = fir.if %[[VAL_10]] -> (i32) {
    ! CHECK:    %[[VAL_14:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
    ! CHECK:    %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_14]] : i32
    ! CHECK:    %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_12]], %[[VAL_14]] : i32
    ! CHECK:    fir.result %[[VAL_16]] : i32
    ! CHECK:  } else {
    ! CHECK:    fir.result %[[VAL_12]] : i32
    ! CHECK:  }
    ! CHECK:  fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[VAL_13]]) : (!fir.ref<i8>, i32) -> i1
    end subroutine 
    
    ! CHECK-LABEL: func @_QMmax_testPdynamic_optional_weird(
    ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
    ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"},
    ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional},
    ! CHECK-SAME:  %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "d"},
    ! CHECK-SAME:  %[[VAL_4:.*]]: !fir.ref<i32> {fir.bindc_name = "e", fir.optional}) {
    subroutine dynamic_optional_weird(a, b, c, d, e)
      integer :: a, b, d
      integer, optional :: c, e
      ! a3, a4, a6, a8 statically missing. a5, a9 dynamically optional.
      print *, max(a1=a, a2=b, a5=c, a7=d, a9 = e)
    ! CHECK:  %[[VAL_10:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
    ! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
    ! CHECK:  %[[VAL_12:.*]] = fir.is_present %[[VAL_2]] : (!fir.ref<i32>) -> i1
    ! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
    ! CHECK:  %[[VAL_14:.*]] = fir.is_present %[[VAL_4]] : (!fir.ref<i32>) -> i1
    ! CHECK:  %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_11]] : i32
    ! CHECK:  %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_10]], %[[VAL_11]] : i32
    ! CHECK:  %[[VAL_17:.*]] = fir.if %[[VAL_12]] -> (i32) {
    ! CHECK:    %[[VAL_18:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
    ! CHECK:    %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_18]] : i32
    ! CHECK:    %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_16]], %[[VAL_18]] : i32
    ! CHECK:    fir.result %[[VAL_20]] : i32
    ! CHECK:  } else {
    ! CHECK:    fir.result %[[VAL_16]] : i32
    ! CHECK:  }
    ! CHECK:  %[[VAL_21:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_13]] : i32
    ! CHECK:  %[[VAL_23:.*]] = arith.select %[[VAL_21]], %[[VAL_17]], %[[VAL_13]] : i32
    ! CHECK:  %[[VAL_24:.*]] = fir.if %[[VAL_14]] -> (i32) {
    ! CHECK:    %[[VAL_25:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
    ! CHECK:    %[[VAL_26:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_25]] : i32
    ! CHECK:    %[[VAL_27:.*]] = arith.select %[[VAL_26]], %[[VAL_23]], %[[VAL_25]] : i32
    ! CHECK:    fir.result %[[VAL_27]] : i32
    ! CHECK:  } else {
    ! CHECK:    fir.result %[[VAL_23]] : i32
    ! CHECK:  }
    ! CHECK:  fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[VAL_24]]) : (!fir.ref<i8>, i32) -> i1
    end subroutine 
    end module
    
      use :: max_test
      integer :: a(4) = [1,12,23, 34]
      integer :: b(4) = [31,22,13, 4]
      integer :: c(4) = [21,32,3, 14]
      call dynamic_optional(a, b)
      call dynamic_optional(a, b, c)
      call dynamic_optional_array_expr_scalar_optional(a, b)
      call dynamic_optional_array_expr_scalar_optional(a, b, c(2))
      call dynamic_optional_scalar(a(2), b(2))
      call dynamic_optional_scalar(a(2), b(2), c(2))
    end
