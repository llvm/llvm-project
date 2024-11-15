! Test lowering of designators to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine array_ref(x, n)
  real :: x(:)
  integer(8) :: n
  print *, x(n)
end subroutine
! CHECK-LABEL: func.func @_QParray_ref(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFarray_refEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFarray_refEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_10:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_9]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>

subroutine char_array_ref(x, n)
  character(*) :: x(:)
  print *, x(10)
end subroutine
! CHECK-LABEL: func.func @_QPchar_array_ref(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_array_refEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_array_refEx"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:  %[[VAL_9:.*]] = fir.box_elesize %[[VAL_3]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:  %[[VAL_10:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_11:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_10]])  typeparams %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>

subroutine char_array_ref_cst_len(x, n)
  character(5) :: x(:)
  print *, x(10)
end subroutine
! CHECK-LABEL: func.func @_QPchar_array_ref_cst_len(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_array_ref_cst_lenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_3:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_3]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_array_ref_cst_lenEx"} : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,5>>>, !fir.box<!fir.array<?x!fir.char<1,5>>>)
! CHECK:  %[[VAL_10:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_11:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_10]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index, index) -> !fir.ref<!fir.char<1,5>>

subroutine array_section(x)
  real :: x(10)
  print *, x(2:8:3)
end subroutine
! CHECK-LABEL: func.func @_QParray_section(
! CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_2]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QFarray_sectionEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:  %[[VAL_9:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_10:.*]] = arith.constant 8 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_12:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_14:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_9]]:%[[VAL_10]]:%[[VAL_11]])  shape %[[VAL_13]] : (!fir.ref<!fir.array<10xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<3xf32>>

subroutine array_section_2(x, n)
  real :: x(:)
  integer(8) :: n
  print *, x(n::3)
end subroutine
! CHECK-LABEL: func.func @_QParray_section_2(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFarray_section_2En"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFarray_section_2Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_3]]#1, %[[VAL_10]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:  %[[VAL_13:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_15:.*]] = arith.subi %[[VAL_11]]#1, %[[VAL_12]] : index
! CHECK:  %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_13]] : index
! CHECK:  %[[VAL_17:.*]] = arith.divsi %[[VAL_16]], %[[VAL_13]] : index
! CHECK:  %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_14]] : index
! CHECK:  %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_17]], %[[VAL_14]] : index
! CHECK:  %[[VAL_20:.*]] = fir.shape %[[VAL_19]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_21:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_12]]:%[[VAL_11]]#1:%[[VAL_13]])  shape %[[VAL_20]] : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>

subroutine char_array_section(x, n)
  character(*) :: x(:)
  print *, x(::3)
end subroutine
! CHECK-LABEL: func.func @_QPchar_array_section(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_array_sectionEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_array_sectionEx"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:  %[[VAL_9:.*]] = fir.box_elesize %[[VAL_3]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_3]]#1, %[[VAL_11]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_13:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_15:.*]] = arith.subi %[[VAL_12]]#1, %[[VAL_10]] : index
! CHECK:  %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_13]] : index
! CHECK:  %[[VAL_17:.*]] = arith.divsi %[[VAL_16]], %[[VAL_13]] : index
! CHECK:  %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_14]] : index
! CHECK:  %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_17]], %[[VAL_14]] : index
! CHECK:  %[[VAL_20:.*]] = fir.shape %[[VAL_19]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_21:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_10]]:%[[VAL_12]]#1:%[[VAL_13]])  shape %[[VAL_20]] typeparams %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>

subroutine char_array_section_cst_len(x, n)
  character(5) :: x(:)
  print *, x(::3)
end subroutine
! CHECK-LABEL: func.func @_QPchar_array_section_cst_len(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_array_section_cst_lenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_3:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_3]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_array_section_cst_lenEx"} : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,5>>>, !fir.box<!fir.array<?x!fir.char<1,5>>>)
! CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_4]]#1, %[[VAL_11]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_13:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_15:.*]] = arith.subi %[[VAL_12]]#1, %[[VAL_10]] : index
! CHECK:  %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_13]] : index
! CHECK:  %[[VAL_17:.*]] = arith.divsi %[[VAL_16]], %[[VAL_13]] : index
! CHECK:  %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_14]] : index
! CHECK:  %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_17]], %[[VAL_14]] : index
! CHECK:  %[[VAL_20:.*]] = fir.shape %[[VAL_19]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_21:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_10]]:%[[VAL_12]]#1:%[[VAL_13]])  shape %[[VAL_20]] typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,5>>>

! Checks related to complex numbers

subroutine complex_imag_ref(x)
  complex :: x(:)
  print *, x%im
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_imag_ref(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFcomplex_imag_refEx"} : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.dscope) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_4:.*]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_2]]#0  imag shape %[[VAL_3]] : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>

subroutine complex_real_ref(x)
  complex :: x(:)
  print *, x%re
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_real_ref(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFcomplex_real_refEx"} : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.dscope) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_4:.*]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_2]]#0  real shape %[[VAL_3]] : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>

subroutine complex_individual_ref(x, n)
  complex :: x(:)
  integer :: n
  print *, x(n)%im
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_individual_ref(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFcomplex_individual_refEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFcomplex_individual_refEx"} : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.dscope) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
! CHECK:  %[[VAL_6:.*]] = hlfir.designate %{{[0-9]+}}#0 (%[[VAL_5]]) imag : (!fir.box<!fir.array<?xcomplex<f32>>>, i64) -> !fir.ref<f32>

subroutine complex_slice_ref(x, start, end)
  complex :: x(:)
  integer :: start, end
  print *, x(start:end)%re
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_slice_ref(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFcomplex_slice_refEend"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFcomplex_slice_refEstart"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %arg0 dummy_scope %{{[0-9]+}} {uniq_name = "_QFcomplex_slice_refEx"} : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.dscope) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i64
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:  %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_9]] : index
! CHECK:  %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %{{.*}} : index
! CHECK:  %[[VAL_13:.*]] = arith.divsi %[[VAL_12]], %{{.*}} : index
! CHECK:  %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %{{.*}} : index
! CHECK:  %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_13]], %{{.*}} : index
! CHECK:  %[[VAL_16:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_17:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]]:%[[VAL_10]]:%{{.*}}) real shape %[[VAL_16]] : (!fir.box<!fir.array<?xcomplex<f32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
