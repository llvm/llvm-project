! Test lowering of inquiry intrinsics with assumed-ranks arguments.
! RUN: bbc -emit-hlfir -o - %s -allow-assumed-rank | FileCheck %s

subroutine test_allocated(x)
  real, allocatable :: x(..)
  call takes_logical(allocated(x))
end subroutine

subroutine test_associated_1(x)
  real, pointer :: x(..)
  call takes_logical(associated(x))
end subroutine

subroutine test_associated_2(x, y)
  real, pointer :: x(..)
  real, target :: y(:)
  call takes_logical(associated(x, y))
end subroutine

subroutine test_associated_3(x, y)
  real, pointer :: x(..)
  real, pointer :: y(..)
  call takes_logical(associated(x, y))
end subroutine

subroutine test_len_1(x)
  character(*) :: x(..)
  call takes_integer(len(x))
end subroutine

subroutine test_len_2(x)
  character(*), pointer :: x(..)
  call takes_integer(len(x))
end subroutine

subroutine test_storage_size_1(x)
  class(*) :: x(..)
  call takes_integer(storage_size(x))
end subroutine

subroutine test_storage_size_2(x)
  class(*), pointer :: x(..)
  call takes_integer(storage_size(x))
end subroutine

subroutine test_present_1(x)
  class(*), optional :: x(..)
  call takes_logical(present(x))
end subroutine

subroutine test_present_2(x)
  class(*), optional, pointer :: x(..)
  call takes_logical(present(x))
end subroutine

subroutine test_is_contiguous_1(x)
  class(*) :: x(..)
  call takes_logical(is_contiguous(x))
end subroutine

subroutine test_is_contiguous_2(x)
  class(*), pointer :: x(..)
  call takes_logical(is_contiguous(x))
end subroutine

subroutine test_same_type_as_1(x, y)
  class(*) :: x(..), y(..)
  call takes_logical(same_type_as(x, y))
end subroutine

subroutine test_same_type_as_2(x, y)
  class(*), pointer :: x(..), y(..)
  call takes_logical(same_type_as(x, y))
end subroutine

subroutine test_extends_type_of_1(x, y)
  class(*) :: x(..), y(..)
  call takes_logical(extends_type_of(x, y))
end subroutine

subroutine test_extends_type_of_2(x, y)
  class(*), pointer :: x(..), y(..)
  call takes_logical(extends_type_of(x, y))
end subroutine

subroutine c_loc_1(x)
  use iso_c_binding, only : c_loc
  real, target :: x(..)
  call takes_cloc(c_loc(x))
end subroutine

subroutine c_loc_2(x)
  use iso_c_binding, only : c_loc
  real, pointer :: x(..)
  call takes_cloc(c_loc(x))
end subroutine

! CHECK-LABEL:   func.func @_QPtest_allocated(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatedEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<*:f32>>>) -> !fir.heap<!fir.array<*:f32>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.array<*:f32>>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_9]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_9]]#1, %[[VAL_9]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_associated_1(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_associated_1Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> !fir.ptr<!fir.array<*:f32>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<!fir.array<*:f32>>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_9]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_9]]#1, %[[VAL_9]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_associated_2(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                    %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y", fir.target}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_associated_2Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest_associated_2Ey"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_6]], %[[VAL_7]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_10:.*]]:3 = hlfir.associate %[[VAL_9]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_10]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_10]]#1, %[[VAL_10]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_associated_3(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                    %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_associated_3Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_associated_3Ey"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_9:.*]] = fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_7]], %[[VAL_8]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_11:.*]]:3 = hlfir.associate %[[VAL_10]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_11]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_11]]#1, %[[VAL_11]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_len_1(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.box<!fir.array<*:!fir.char<1,?>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_len_1Ex"} : (!fir.box<!fir.array<*:!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<*:!fir.char<1,?>>>, !fir.box<!fir.array<*:!fir.char<1,?>>>)
! CHECK:           %[[VAL_3:.*]] = fir.box_elesize %[[VAL_2]]#0 : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (index) -> i32
! CHECK:           %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPtakes_integer(%[[VAL_5]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_len_2(
! CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<*:!fir.char<1,?>>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<*:!fir.char<1,?>>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_elesize %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<*:!fir.char<1,?>>>>) -> index
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_3]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_len_2Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:!fir.char<1,?>>>>>, index, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<*:!fir.char<1,?>>>>>)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (index) -> i32
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPtakes_integer(%[[VAL_6]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_storage_size_1(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_storage_size_1Ex"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_3:.*]] = fir.box_elesize %[[VAL_2]]#0 : (!fir.class<!fir.array<*:none>>) -> i32
! CHECK:           %[[VAL_4:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_3]], %[[VAL_4]] : i32
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPtakes_integer(%[[VAL_6]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_storage_size_2(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_storage_size_2Ex"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.class<!fir.ptr<!fir.array<*:none>>>) -> !fir.ptr<!fir.array<*:none>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<!fir.array<*:none>>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           fir.if %[[VAL_7]] {
! CHECK:             %[[VAL_13:.*]] = fir.call @_FortranAReportFatalUserError
! CHECK:           }
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>
! CHECK:           %[[VAL_15:.*]] = fir.box_elesize %[[VAL_14]] : (!fir.class<!fir.ptr<!fir.array<*:none>>>) -> i32
! CHECK:           %[[VAL_16:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_15]], %[[VAL_16]] : i32
! CHECK:           %[[VAL_18:.*]]:3 = hlfir.associate %[[VAL_17]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QPtakes_integer(%[[VAL_18]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_18]]#1, %[[VAL_18]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_present_1(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_present_1Ex"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_3:.*]] = fir.is_present %[[VAL_2]]#0 : (!fir.class<!fir.array<*:none>>) -> i1
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_5]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_present_2(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<optional, pointer>, uniq_name = "_QFtest_present_2Ex"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.is_present %[[VAL_2]]#0 : (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>) -> i1
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_5]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_is_contiguous_1(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_is_contiguous_1Ex"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.class<!fir.array<*:none>>) -> !fir.box<none>
! CHECK:           %[[VAL_4:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_3]]) fastmath<contract> : (!fir.box<none>) -> i1
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_6]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_is_contiguous_2(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_is_contiguous_2Ex"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.class<!fir.ptr<!fir.array<*:none>>>) -> !fir.box<none>
! CHECK:           %[[VAL_5:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_4]]) fastmath<contract> : (!fir.box<none>) -> i1
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_6]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_7]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_same_type_as_1(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "x"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_same_type_as_1Ex"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_same_type_as_1Ey"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.class<!fir.array<*:none>>) -> !fir.box<none>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.class<!fir.array<*:none>>) -> !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.call @_FortranASameTypeAs(%[[VAL_5]], %[[VAL_6]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_9]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_9]]#1, %[[VAL_9]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_same_type_as_2(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_same_type_as_2Ex"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_same_type_as_2Ey"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (!fir.class<!fir.ptr<!fir.array<*:none>>>) -> !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (!fir.class<!fir.ptr<!fir.array<*:none>>>) -> !fir.box<none>
! CHECK:           %[[VAL_9:.*]] = fir.call @_FortranASameTypeAs(%[[VAL_7]], %[[VAL_8]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_11:.*]]:3 = hlfir.associate %[[VAL_10]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_11]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_11]]#1, %[[VAL_11]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_extends_type_of_1(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "x"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_extends_type_of_1Ex"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_extends_type_of_1Ey"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.class<!fir.array<*:none>>) -> !fir.box<none>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.class<!fir.array<*:none>>) -> !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.call @_FortranAExtendsTypeOf(%[[VAL_5]], %[[VAL_6]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_9]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_9]]#1, %[[VAL_9]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_extends_type_of_2(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_extends_type_of_2Ex"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_extends_type_of_2Ey"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<*:none>>>>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (!fir.class<!fir.ptr<!fir.array<*:none>>>) -> !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (!fir.class<!fir.ptr<!fir.array<*:none>>>) -> !fir.box<none>
! CHECK:           %[[VAL_9:.*]] = fir.call @_FortranAExtendsTypeOf(%[[VAL_7]], %[[VAL_8]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_11:.*]]:3 = hlfir.associate %[[VAL_10]] {adapt.valuebyref} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           fir.call @_QPtakes_logical(%[[VAL_11]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_11]]#1, %[[VAL_11]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPc_loc_1(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_1Ex"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_6:.*]] = fir.box_addr %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.ref<!fir.array<*:f32>>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.array<*:f32>>) -> i64
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_5]] : !fir.ref<i64>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant false
! CHECK:           %[[VAL_10:.*]] = hlfir.as_expr %[[VAL_8]]#0 move %[[VAL_9]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, i1) -> !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:           %[[VAL_11:.*]]:3 = hlfir.associate %[[VAL_10]] {adapt.valuebyref} : (!hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, i1)
! CHECK:           fir.call @_QPtakes_cloc(%[[VAL_11]]#1) fastmath<contract> : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_11]]#1, %[[VAL_11]]#2 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, i1
! CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPc_loc_2(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFc_loc_2Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_5:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_5]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_7:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> !fir.ptr<!fir.array<*:f32>>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ptr<!fir.array<*:f32>>) -> i64
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_6]] : !fir.ref<i64>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant false
! CHECK:           %[[VAL_11:.*]] = hlfir.as_expr %[[VAL_9]]#0 move %[[VAL_10]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, i1) -> !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:           %[[VAL_12:.*]]:3 = hlfir.associate %[[VAL_11]] {adapt.valuebyref} : (!hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, i1)
! CHECK:           fir.call @_QPtakes_cloc(%[[VAL_12]]#1) fastmath<contract> : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_12]]#1, %[[VAL_12]]#2 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, i1
! CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:           return
! CHECK:         }
