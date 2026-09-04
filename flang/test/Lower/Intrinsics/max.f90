! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module max_test
    contains
    ! CHECK-LABEL: func.func @_QMmax_testPdynamic_optional(
    ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
    ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"},
    ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c", fir.optional}) {
    subroutine dynamic_optional(a, b, c)
      integer :: a(:), b(:)
      integer, optional :: c(:)
    ! CHECK-DAG:  %[[A:.*]]:2 = hlfir.declare %[[VAL_0]]
    ! CHECK-DAG:  %[[B:.*]]:2 = hlfir.declare %[[VAL_1]]
    ! CHECK-DAG:  %[[C:.*]]:2 = hlfir.declare %[[VAL_2]]
    ! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[C]]#0 : (!fir.box<!fir.array<?xi32>>) -> i1
    ! CHECK:  %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
    ! CHECK:  hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ! CHECK:    %[[A_ELT:.*]] = fir.load %{{.*}} : !fir.ref<i32>
    ! CHECK:    %[[B_ELT:.*]] = fir.load %{{.*}} : !fir.ref<i32>
    ! CHECK:    %[[MAX_AB:.*]] = arith.maxsi %[[A_ELT]], %[[B_ELT]] : i32
    ! CHECK:    fir.if %[[IS_PRESENT]] -> (i32) {
    ! CHECK:      %[[C_ELT:.*]] = fir.load %{{.*}} : !fir.ref<i32>
    ! CHECK:      arith.maxsi %[[MAX_AB]], %[[C_ELT]] : i32
    ! CHECK:    }
    ! CHECK:  }
    ! CHECK:  OutputDescriptor
    ! CHECK:  EndIoStatement
      print *, max(a, b, c)
    end subroutine

    ! CHECK-LABEL: func.func @_QMmax_testPdynamic_optional_array_expr_scalar_optional(
    ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
    ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"},
    ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional}) {
    subroutine dynamic_optional_array_expr_scalar_optional(a, b, c)
      integer :: a(:), b(:)
      integer, optional :: c
      print *, max(a, b, c)
    ! CHECK-DAG:  %[[A:.*]]:2 = hlfir.declare %[[VAL_0]]
    ! CHECK-DAG:  %[[B:.*]]:2 = hlfir.declare %[[VAL_1]]
    ! CHECK-DAG:  %[[C:.*]]:2 = hlfir.declare %[[VAL_2]]
    ! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[C]]#0 : (!fir.ref<i32>) -> i1
    ! CHECK:  hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ! CHECK:    %[[A_ELT:.*]] = fir.load %{{.*}} : !fir.ref<i32>
    ! CHECK:    %[[B_ELT:.*]] = fir.load %{{.*}} : !fir.ref<i32>
    ! CHECK:    %[[MAX_AB:.*]] = arith.maxsi %[[A_ELT]], %[[B_ELT]] : i32
    ! CHECK:    fir.if %[[IS_PRESENT]] -> (i32) {
    ! CHECK:      %[[C_VAL:.*]] = fir.load %[[C]]#0 : !fir.ref<i32>
    ! CHECK:      arith.maxsi %[[MAX_AB]], %[[C_VAL]] : i32
    ! CHECK:    }
    ! CHECK:  }
    ! CHECK:  OutputDescriptor
    ! CHECK:  EndIoStatement
    end subroutine

    ! CHECK-LABEL: func.func @_QMmax_testPdynamic_optional_scalar(
    ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
    ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"},
    ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "c", fir.optional}) {
    subroutine dynamic_optional_scalar(a, b, c)
      integer :: a, b
      integer, optional :: c
      print *, max(a, b, c)
    ! CHECK-DAG:  %[[A:.*]]:2 = hlfir.declare %[[VAL_0]]
    ! CHECK-DAG:  %[[B:.*]]:2 = hlfir.declare %[[VAL_1]]
    ! CHECK-DAG:  %[[C:.*]]:2 = hlfir.declare %[[VAL_2]]
    ! CHECK:  %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
    ! CHECK:  %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
    ! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[C]]#0 : (!fir.ref<i32>) -> i1
    ! CHECK:  %[[MAX_AB:.*]] = arith.maxsi %[[A_VAL]], %[[B_VAL]] : i32
    ! CHECK:  %[[RESULT:.*]] = fir.if %[[IS_PRESENT]] -> (i32) {
    ! CHECK:    %[[C_VAL:.*]] = fir.load %[[C]]#0 : !fir.ref<i32>
    ! CHECK:    %[[MAX_ABC:.*]] = arith.maxsi %[[MAX_AB]], %[[C_VAL]] : i32
    ! CHECK:    fir.result %[[MAX_ABC]] : i32
    ! CHECK:  } else {
    ! CHECK:    fir.result %[[MAX_AB]] : i32
    ! CHECK:  }
    ! CHECK:  fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[RESULT]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    end subroutine

    ! CHECK-LABEL: func.func @_QMmax_testPdynamic_optional_weird(
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
    ! CHECK-DAG:  %[[A:.*]]:2 = hlfir.declare %[[VAL_0]]
    ! CHECK-DAG:  %[[B:.*]]:2 = hlfir.declare %[[VAL_1]]
    ! CHECK-DAG:  %[[C:.*]]:2 = hlfir.declare %[[VAL_2]]
    ! CHECK-DAG:  %[[D:.*]]:2 = hlfir.declare %[[VAL_3]]
    ! CHECK-DAG:  %[[E:.*]]:2 = hlfir.declare %[[VAL_4]]
    ! CHECK:  %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
    ! CHECK:  %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
    ! CHECK:  %[[IS_C:.*]] = fir.is_present %[[C]]#0 : (!fir.ref<i32>) -> i1
    ! CHECK:  %[[D_VAL:.*]] = fir.load %[[D]]#0 : !fir.ref<i32>
    ! CHECK:  %[[IS_E:.*]] = fir.is_present %[[E]]#0 : (!fir.ref<i32>) -> i1
    ! CHECK:  %[[MAX_AB:.*]] = arith.maxsi %[[A_VAL]], %[[B_VAL]] : i32
    ! CHECK:  %[[MAX_ABC:.*]] = fir.if %[[IS_C]] -> (i32) {
    ! CHECK:    %[[C_VAL:.*]] = fir.load %[[C]]#0 : !fir.ref<i32>
    ! CHECK:    %[[R:.*]] = arith.maxsi %[[MAX_AB]], %[[C_VAL]] : i32
    ! CHECK:    fir.result %[[R]] : i32
    ! CHECK:  } else {
    ! CHECK:    fir.result %[[MAX_AB]] : i32
    ! CHECK:  }
    ! CHECK:  %[[MAX_ABCD:.*]] = arith.maxsi %[[MAX_ABC]], %[[D_VAL]] : i32
    ! CHECK:  %[[MAX_ABCDE:.*]] = fir.if %[[IS_E]] -> (i32) {
    ! CHECK:    %[[E_VAL:.*]] = fir.load %[[E]]#0 : !fir.ref<i32>
    ! CHECK:    %[[R:.*]] = arith.maxsi %[[MAX_ABCD]], %[[E_VAL]] : i32
    ! CHECK:    fir.result %[[R]] : i32
    ! CHECK:  } else {
    ! CHECK:    fir.result %[[MAX_ABCD]] : i32
    ! CHECK:  }
    ! CHECK:  fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[MAX_ABCDE]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
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
