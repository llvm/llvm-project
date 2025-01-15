! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Addresses assumed shape dummy argument with VALUE keyword

subroutine test_integer_value1(x)
  integer, value :: x(:)
  call internal_call1(x)
end

! CHECK-LABEL:  func.func @_QPtest_integer_value1(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_integer_value1Ex"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:          %[[VAL_1:.*]]:2 = hlfir.copy_in %[[VAL_0]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.box<!fir.array<?xi32>>, i1)
! CHECK:          %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:          fir.call @_QPinternal_call1(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>) -> ()
! CHECK:          hlfir.copy_out %[[TMP_BOX]], %[[VAL_1]]#1 to %[[VAL_0]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i1, !fir.box<!fir.array<?xi32>>) -> ()
! CHECK:          return
! CHECK:        }

subroutine test_integer_value2(x)
  integer, value :: x(:,:)
  call internal_call2(x)
end
! CHECK-LABEL:  func.func @_QPtest_integer_value2(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "x"}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_integer_value2Ex"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
! CHECK:          %[[VAL_1:.*]]:2 = hlfir.copy_in %[[VAL_0]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> (!fir.box<!fir.array<?x?xi32>>, i1)
! CHECK:          %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#0 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.ref<!fir.array<?x?xi32>>
! CHECK:          fir.call @_QPinternal_call2(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.array<?x?xi32>>) -> ()
! CHECK:          hlfir.copy_out %[[TMP_BOX]], %[[VAL_1]]#1 to %[[VAL_0]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, i1, !fir.box<!fir.array<?x?xi32>>) -> ()
! CHECK:          return
! CHECK:        }

subroutine test_real_value1(x) 
  real, value :: x(:)
  call internal_call3(x)
end
! CHECK-LABEL:  func.func @_QPtest_real_value1(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_real_value1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:          %[[VAL_1:.*]]:2 = hlfir.copy_in %[[VAL_0]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
! CHECK:          %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:          fir.call @_QPinternal_call3(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:          hlfir.copy_out %[[TMP_BOX]], %[[VAL_1]]#1 to %[[VAL_0]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
! CHECK:          return
! CHECK:        }

subroutine test_real_value2(x) 
  real, value :: x(:,:)
  call internal_call4(x)
end
! CHECK-LABEL:  func.func @_QPtest_real_value2(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_real_value2Ex"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:          %[[VAL_1:.*]]:2 = hlfir.copy_in %[[VAL_0]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> (!fir.box<!fir.array<?x?xf32>>, i1)
! CHECK:          %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#0 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:          fir.call @_QPinternal_call4(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.array<?x?xf32>>) -> ()
! CHECK:          hlfir.copy_out %[[TMP_BOX]], %[[VAL_1]]#1 to %[[VAL_0]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, i1, !fir.box<!fir.array<?x?xf32>>) -> ()
! CHECK:          return
! CHECK:        }

subroutine test_complex_value1(x)
  complex, value :: x(:)
  call internal_call5(x)
end
! CHECK-LABEL:  func.func @_QPtest_complex_value1(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?xcomplex<f32>>> {fir.bindc_name = "x"}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_complex_value1Ex"} : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.dscope) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
! CHECK:          %[[VAL_1:.*]]:2 = hlfir.copy_in %[[VAL_0]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>) -> (!fir.box<!fir.array<?xcomplex<f32>>>, i1)
! CHECK:          %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#0 : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.ref<!fir.array<?xcomplex<f32>>>
! CHECK:          fir.call @_QPinternal_call5(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.array<?xcomplex<f32>>>) -> ()
! CHECK:          hlfir.copy_out %[[TMP_BOX]], %[[VAL_1]]#1 to %[[VAL_0]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>, i1, !fir.box<!fir.array<?xcomplex<f32>>>) -> ()
! CHECK:          return
! CHECK:        }

subroutine test_complex_value2(x)
  complex, value :: x(:,:)
  call internal_call6(x)
end
! CHECK-LABEL:  func.func @_QPtest_complex_value2(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?x?xcomplex<f32>>> {fir.bindc_name = "x"}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_complex_value2Ex"} : (!fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.box<!fir.array<?x?xcomplex<f32>>>)
! CHECK:          %[[VAL_1:.*]]:2 = hlfir.copy_in %[[VAL_0]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xcomplex<f32>>>>>) -> (!fir.box<!fir.array<?x?xcomplex<f32>>>, i1)
! CHECK:          %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#0 : (!fir.box<!fir.array<?x?xcomplex<f32>>>) -> !fir.ref<!fir.array<?x?xcomplex<f32>>>
! CHECK:          fir.call @_QPinternal_call6(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.array<?x?xcomplex<f32>>>) -> ()
! CHECK:          hlfir.copy_out %[[TMP_BOX]], %[[VAL_1]]#1 to %[[VAL_0]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xcomplex<f32>>>>>, i1, !fir.box<!fir.array<?x?xcomplex<f32>>>) -> ()
! CHECK:          return
! CHECK:        }

subroutine test_optional1(x)
  real, value, optional :: x(:)
  if (present(x)) then
    call internal_call7(x)
  endif
end
! CHECK-LABEL:  func.func @_QPtest_optional1(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional, value>, uniq_name = "_QFtest_optional1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:          %[[VAL_1:.*]] = fir.is_present %[[VAL_0]]#1 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:          fir.if %[[VAL_1:.*]] {
! CHECK:            %[[VAL_2:.*]]:2 = hlfir.copy_in %[[VAL_0]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
! CHECK:            %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:            fir.call @_QPinternal_call7(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:            hlfir.copy_out %[[TMP_BOX]], %[[VAL_2]]#1 to %[[VAL_0]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
! CHECK:          }
! CHECK:          return
! CHECK:        }

subroutine test_optional2(x)
  real, value, optional :: x(:,:)
  if (present(x)) then
    call internal_call8(x)
  endif
end
! CHECK-LABEL:  func.func @_QPtest_optional2(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional, value>, uniq_name = "_QFtest_optional2Ex"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:          %[[VAL_1:.*]] = fir.is_present %[[VAL_0]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> i1
! CHECK:          fir.if %[[VAL_1:.*]] {
! CHECK:            %[[VAL_2:.*]]:2 = hlfir.copy_in %[[VAL_0]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> (!fir.box<!fir.array<?x?xf32>>, i1)
! CHECK:            %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]]#0 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:            fir.call @_QPinternal_call8(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.array<?x?xf32>>) -> ()
! CHECK:            hlfir.copy_out %[[TMP_BOX]], %[[VAL_2]]#1 to %[[VAL_0]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, i1, !fir.box<!fir.array<?x?xf32>>) -> ()
! CHECK:          }
! CHECK:          return
! CHECK:        }

subroutine test_optional3(x)
  real, value, optional :: x(:)
  if (present(x)) then
    stop
  endif
end
! CHECK-LABEL:  func.func @_QPtest_optional3(
! CHECK-SAME:     %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:          %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional, value>, uniq_name = "_QFtest_optional3Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:          %[[VAL_1:.*]] = fir.is_present %[[VAL_0]]#1 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:          cf.cond_br %[[VAL_1]], ^bb1, ^bb2
! CHECK:          b1:  // pred: ^bb0
! CHECK:          %[[C0_I32:.*]] = arith.constant 0 : i32
! CHECK:          %[[FALSE:.*]] = arith.constant false
! CHECK:          %[[FALSE_0:.*]] = arith.constant false
! CHECK:          %[[VAL_2:.*]] = fir.call @_FortranAStopStatement(%[[C0_I32]], %[[FALSE]], %[[FALSE]]_0) fastmath<contract> : (i32, i1, i1) -> none
! CHECK:          fir.unreachable
! CHECK:          b2:  // pred: ^bb0
! CHECK:          return
! CHECK:        }
