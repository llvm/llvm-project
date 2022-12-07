! Test lowering of variables to fir.declare
! RUN: bbc -emit-fir -hlfir %s -o - | FileCheck %s

subroutine scalar_numeric(x)
  integer :: x
end subroutine
! CHECK-LABEL: func.func @_QPscalar_numeric(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>
! CHECK:  %[[VAL_1:.*]] = hlfir.declare %[[VAL_0]] {uniq_name = "_QFscalar_numericEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine scalar_character(c)
  character(*) :: c
end subroutine
! CHECK-LABEL: func.func @_QPscalar_character(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_2:.*]] = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 {uniq_name = "_QFscalar_characterEc"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)

subroutine scalar_character_cst_len(c)
  character(10) :: c
end subroutine
! CHECK-LABEL: func.func @_QPscalar_character_cst_len(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_2]] {uniq_name = "_QFscalar_character_cst_lenEc"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)

subroutine array_numeric(x)
  integer :: x(10, 20)
end subroutine
! CHECK-LABEL: func.func @_QParray_numeric(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x20xi32>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_2:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_4:.*]] = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "_QFarray_numericEx"} : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x20xi32>>, !fir.ref<!fir.array<10x20xi32>>)


subroutine array_numeric_lbounds(x)
  integer :: x(-1:10, -2:20)
end subroutine
! CHECK-LABEL: func.func @_QParray_numeric_lbounds(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<12x23xi32>>
! CHECK:  %[[VAL_1:.*]] = arith.constant -1 : index
! CHECK:  %[[VAL_2:.*]] = arith.constant 12 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant -2 : index
! CHECK:  %[[VAL_4:.*]] = arith.constant 23 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:  %[[VAL_6:.*]] = hlfir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "_QFarray_numeric_lboundsEx"} : (!fir.ref<!fir.array<12x23xi32>>, !fir.shapeshift<2>) -> (!fir.box<!fir.array<12x23xi32>>, !fir.ref<!fir.array<12x23xi32>>)

subroutine array_character(c)
  character(*) :: c(50)
end subroutine
! CHECK-LABEL: func.func @_QParray_character(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<50x!fir.char<1,?>>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 50 : index
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = hlfir.declare %[[VAL_2]](%[[VAL_4]]) typeparams %[[VAL_1]]#1 {uniq_name = "_QFarray_characterEc"} : (!fir.ref<!fir.array<50x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<50x!fir.char<1,?>>>, !fir.ref<!fir.array<50x!fir.char<1,?>>>)

subroutine scalar_numeric_attributes(x)
  integer, optional, target, intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPscalar_numeric_attributes(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>
! CHECK:  %[[VAL_1:.*]] = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in, optional, target>, uniq_name = "_QFscalar_numeric_attributesEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine scalar_numeric_attributes_2(x)
  real(16), value :: x(100)
end subroutine
! CHECK-LABEL: func.func @_QPscalar_numeric_attributes_2(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf128>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_3:.*]] = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFscalar_numeric_attributes_2Ex"} : (!fir.ref<!fir.array<100xf128>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf128>>, !fir.ref<!fir.array<100xf128>>)

subroutine scalar_numeric_attributes_3(x)
  real, intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPscalar_numeric_attributes_3(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<f32>
! CHECK:  %[[VAL_1:.*]] = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFscalar_numeric_attributes_3Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

subroutine scalar_numeric_attributes_4(x)
  logical(8), intent(out) :: x
end subroutine
! CHECK-LABEL: func.func @_QPscalar_numeric_attributes_4(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.logical<8>>
! CHECK:  %[[VAL_1:.*]] = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFscalar_numeric_attributes_4Ex"} : (!fir.ref<!fir.logical<8>>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>)

subroutine scalar_numeric_parameter()
  integer, parameter :: p = 42
end subroutine
! CHECK-LABEL: func.func @_QPscalar_numeric_parameter() {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QFscalar_numeric_parameterECp) : !fir.ref<i32>
! CHECK:  %[[VAL_1:.*]] = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFscalar_numeric_parameterECp"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
