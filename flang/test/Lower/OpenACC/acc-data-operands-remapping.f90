! Test remapping of variables appearing in OpenACC data clauses
! to the related acc dialect data operation result.

! This tests checks how the hlfir.declare is recreated and used inside
! the acc compute region.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module m
interface
subroutine takes_scalar(x)
     real :: x
end subroutine
subroutine takes_scalar_character(c, l)
     integer :: l
     character(l) :: c
end subroutine
subroutine takes_explicit_cst_shape(x)
     real :: x(100)
end subroutine
subroutine takes_explicit_shape(x, n)
     real :: x(n)
end subroutine
subroutine takes_assumed_shape(x)
     real :: x(:)
end subroutine
subroutine takes_pointer(x)
     real, pointer :: x(:)
end subroutine

subroutine takes_optional_scalar(x)
     real, optional :: x
end subroutine
subroutine takes_optional_explicit_cst_shape(x)
     real, optional :: x(100)
end subroutine
subroutine takes_optional_explicit_shape(x, n)
     real, optional :: x(n)
end subroutine
subroutine takes_optional_assumed_shape(x)
     real, optional :: x(:)
end subroutine
subroutine takes_optional_pointer(x)
     real, optional, pointer :: x(:)
end subroutine
end interface
contains

! ----------------------------- Test forwarding ------------------------------ !

   subroutine test_scalar(x)
     real :: x
     !$acc parallel copy(x)
        call takes_scalar(x)
     !$acc end parallel
   end subroutine

   subroutine test_scalar_character(c, l)
      integer :: l
      character(l) :: c
     !$acc parallel copy(x)
        call takes_scalar_character(c, len(c))
     !$acc end parallel
   end subroutine

   subroutine test_cst_shape(x)
     real :: x(100)
     !$acc parallel copy(x)
        call takes_explicit_cst_shape(x)
     !$acc end parallel
   end subroutine

   subroutine test_explicit_shape(x, n)
     real :: x(n)
     !$acc parallel copy(x)
        call takes_explicit_shape(x, size(x,dim=1))
     !$acc end parallel
   end subroutine

   subroutine test_assumed_shape(x, n)
     real :: x(:)
     !$acc parallel copy(x)
        call takes_assumed_shape(x)
     !$acc end parallel
   end subroutine

   subroutine test_contiguous_assumed_shape(x, n)
     real, contiguous :: x(:)
     !$acc parallel copy(x)
        call takes_explicit_shape(x, size(x,dim=1))
     !$acc end parallel
   end subroutine

   subroutine test_pointer(x, n)
     real, pointer :: x(:)
     !$acc parallel copy(x)
        call takes_pointer(x)
     !$acc end parallel
   end subroutine

   subroutine test_using_both_results(x, n)
     real :: x(n)
     !$acc parallel copy(x)
        ! using hlfir.declare result #0
        call takes_assumed_shape(x)
        ! using hlfir.declare result #1
        call takes_explicit_shape(x, size(x,dim=1))
     !$acc end parallel
   end subroutine

! ------------------------- Test array addressing ---------------------------- !

   subroutine addressing_cst_shape(x)
     real :: x(10, 20)
     !$acc parallel copy(x)
        call takes_scalar(x(2,3))
     !$acc end parallel
   end subroutine

   subroutine addressing_explicit_shape(x, n, m)
     real :: x(n, m)
     !$acc parallel copy(x)
        call takes_scalar(x(2,3))
     !$acc end parallel
   end subroutine

   subroutine addressing_assumed_shape(x, n)
     real :: x(:, :)
     !$acc parallel copy(x)
        call takes_scalar(x(2,3))
     !$acc end parallel
   end subroutine

   subroutine addressing_contiguous_assumed_shape(x, n)
     real, contiguous :: x(:, :)
     !$acc parallel copy(x)
        call takes_scalar(x(2,3))
     !$acc end parallel
   end subroutine

   subroutine addressing_pointer(x)
     real, pointer :: x(:, :)
     !$acc parallel copy(x)
        call takes_scalar(x(2,3))
     !$acc end parallel
   end subroutine

! ------------------------ Test OPTIONAL handling ---------------------------- !

   subroutine test_optional_scalar(x)
     real, optional :: x
     !$acc parallel copy(x)
        call takes_optional_scalar(x)
     !$acc end parallel
   end subroutine

   subroutine test_optional_explicit_cst_shape(x)
      real, optional :: x(100)
     !$acc parallel copy(x)
        call takes_optional_explicit_cst_shape(x)
     !$acc end parallel
   end subroutine

   subroutine test_optional_explicit_shape(x, n)
      real, optional :: x(n)
      !$acc parallel copy(x)
        call takes_optional_explicit_shape(x, n)
      !$acc end parallel
   end subroutine

   subroutine test_optional_assumed_shape(x)
      real, optional :: x(:)
      !$acc parallel copy(x)
        call takes_optional_assumed_shape(x)
      !$acc end parallel
   end subroutine

   subroutine test_optional_pointer(x)
      real, optional, pointer :: x(:)
      !$acc parallel copy(x)
        call takes_optional_pointer(x)
      !$acc end parallel
   end subroutine

end module

! CHECK-LABEL:   func.func @_QMmPtest_scalar(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_scalarEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_2:.*]] = acc.copyin varPtr(%[[VAL_1]]#0 : !fir.ref<f32>) -> !fir.ref<f32> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_2]] : !fir.ref<f32>) {
! CHECK:             %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] {uniq_name = "_QMmFtest_scalarEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             fir.call @_QPtakes_scalar(%[[VAL_4]]#0) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_2]] : !fir.ref<f32>) to varPtr(%[[VAL_1]]#0 : !fir.ref<f32>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_scalar_character(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "l"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_scalar_characterEl"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QMmFtest_scalar_characterEx"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QMmFtest_scalar_characterEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_8]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_scalar_characterEc"} : (!fir.ref<!fir.char<1,?>>, i32, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_10:.*]] = acc.copyin varPtr(%[[VAL_3]]#0 : !fir.ref<f32>) -> !fir.ref<f32> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_10]] : !fir.ref<f32>) {
! CHECK:             %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QMmFtest_scalar_characterEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             %[[VAL_12:.*]]:3 = hlfir.associate %[[VAL_8]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:             fir.call @_QPtakes_scalar_character(%[[VAL_9]]#0, %[[VAL_12]]#0) fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:             hlfir.end_associate %[[VAL_12]]#1, %[[VAL_12]]#2 : !fir.ref<i32>, i1
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_10]] : !fir.ref<f32>) to varPtr(%[[VAL_3]]#0 : !fir.ref<f32>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_cst_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_2]]) dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_cst_shapeEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:           %[[VAL_4:.*]] = acc.copyin varPtr(%[[VAL_3]]#0 : !fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_4]] : !fir.ref<!fir.array<100xf32>>) {
! CHECK:             %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_2]]) dummy_scope %[[VAL_5]] {uniq_name = "_QMmFtest_cst_shapeEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:             fir.call @_QPtakes_explicit_cst_shape(%[[VAL_6]]#0) fastmath<contract> : (!fir.ref<!fir.array<100xf32>>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_4]] : !fir.ref<!fir.array<100xf32>>) to varPtr(%[[VAL_3]]#0 : !fir.ref<!fir.array<100xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_explicit_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_explicit_shapeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_8]]) dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_explicit_shapeEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           %[[VAL_10:.*]] = acc.copyin var(%[[VAL_9]]#0 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_10]] : !fir.box<!fir.array<?xf32>>) {
! CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:             %[[VAL_12:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_8]]) dummy_scope %[[VAL_12]] {uniq_name = "_QMmFtest_explicit_shapeEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (index) -> i64
! CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> i32
! CHECK:             %[[VAL_16:.*]]:3 = hlfir.associate %[[VAL_15]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:             fir.call @_QPtakes_explicit_shape(%[[VAL_13]]#1, %[[VAL_16]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>, !fir.ref<i32>) -> ()
! CHECK:             hlfir.end_associate %[[VAL_16]]#1, %[[VAL_16]]#2 : !fir.ref<i32>, i1
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accVar(%[[VAL_10]] : !fir.box<!fir.array<?xf32>>) to var(%[[VAL_9]]#0 : !fir.box<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_assumed_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_assumed_shapeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_assumed_shapeEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_3:.*]] = acc.copyin var(%[[VAL_2]]#0 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_3]] : !fir.box<!fir.array<?xf32>>) {
! CHECK:             %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]] dummy_scope %[[VAL_4]] skip_rebox {uniq_name = "_QMmFtest_assumed_shapeEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:             fir.call @_QPtakes_assumed_shape(%[[VAL_5]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accVar(%[[VAL_3]] : !fir.box<!fir.array<?xf32>>) to var(%[[VAL_2]]#0 : !fir.box<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_contiguous_assumed_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_contiguous_assumed_shapeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[ARG0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]], %[[VAL_4]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_6]]) dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QMmFtest_contiguous_assumed_shapeEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           %[[VAL_8:.*]] = acc.copyin var(%[[VAL_7]]#0 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_8]] : !fir.box<!fir.array<?xf32>>) {
! CHECK:             %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:             %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_6]]) dummy_scope %[[VAL_10]] {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QMmFtest_contiguous_assumed_shapeEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:             %[[VAL_12:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
! CHECK:             %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> i32
! CHECK:             %[[VAL_14:.*]]:3 = hlfir.associate %[[VAL_13]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:             fir.call @_QPtakes_explicit_shape(%[[VAL_11]]#1, %[[VAL_14]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>, !fir.ref<i32>) -> ()
! CHECK:             hlfir.end_associate %[[VAL_14]]#1, %[[VAL_14]]#2 : !fir.ref<i32>, i1
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accVar(%[[VAL_8]] : !fir.box<!fir.array<?xf32>>) to var(%[[VAL_7]]#0 : !fir.box<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_pointer(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmFtest_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_3:.*]] = acc.copyin varPtr(%[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
! CHECK:             %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]] dummy_scope %[[VAL_4]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmFtest_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:             fir.call @_QPtakes_pointer(%[[VAL_5]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) to varPtr(%[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_using_both_results(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_using_both_resultsEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_8]]) dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_using_both_resultsEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           %[[VAL_10:.*]] = acc.copyin var(%[[VAL_9]]#0 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_10]] : !fir.box<!fir.array<?xf32>>) {
! CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:             %[[VAL_12:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_8]]) dummy_scope %[[VAL_12]] {uniq_name = "_QMmFtest_using_both_resultsEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:             fir.call @_QPtakes_assumed_shape(%[[VAL_13]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (index) -> i64
! CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> i32
! CHECK:             %[[VAL_16:.*]]:3 = hlfir.associate %[[VAL_15]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:             fir.call @_QPtakes_explicit_shape(%[[VAL_13]]#1, %[[VAL_16]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>, !fir.ref<i32>) -> ()
! CHECK:             hlfir.end_associate %[[VAL_16]]#1, %[[VAL_16]]#2 : !fir.ref<i32>, i1
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accVar(%[[VAL_10]] : !fir.box<!fir.array<?xf32>>) to var(%[[VAL_9]]#0 : !fir.box<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPaddressing_cst_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10x20xf32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 20 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_3]]) dummy_scope %[[VAL_0]] {uniq_name = "_QMmFaddressing_cst_shapeEx"} : (!fir.ref<!fir.array<10x20xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<10x20xf32>>, !fir.ref<!fir.array<10x20xf32>>)
! CHECK:           %[[VAL_5:.*]] = acc.copyin varPtr(%[[VAL_4]]#0 : !fir.ref<!fir.array<10x20xf32>>) -> !fir.ref<!fir.array<10x20xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_5]] : !fir.ref<!fir.array<10x20xf32>>) {
! CHECK:             %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_3]]) dummy_scope %[[VAL_6]] {uniq_name = "_QMmFaddressing_cst_shapeEx"} : (!fir.ref<!fir.array<10x20xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<10x20xf32>>, !fir.ref<!fir.array<10x20xf32>>)
! CHECK:             %[[VAL_8:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_9:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_8]], %[[VAL_9]])  : (!fir.ref<!fir.array<10x20xf32>>, index, index) -> !fir.ref<f32>
! CHECK:             fir.call @_QPtakes_scalar(%[[VAL_10]]) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_5]] : !fir.ref<!fir.array<10x20xf32>>) to varPtr(%[[VAL_4]]#0 : !fir.ref<!fir.array<10x20xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPaddressing_explicit_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "m"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFaddressing_explicit_shapeEm"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFaddressing_explicit_shapeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> i64
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : index
! CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : index
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : index
! CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_11]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_8]], %[[VAL_14]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_15]]) dummy_scope %[[VAL_0]] {uniq_name = "_QMmFaddressing_explicit_shapeEx"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
! CHECK:           %[[VAL_17:.*]] = acc.copyin var(%[[VAL_16]]#0 : !fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_17]] : !fir.box<!fir.array<?x?xf32>>) {
! CHECK:             %[[VAL_18:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:             %[[VAL_19:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_18]](%[[VAL_15]]) dummy_scope %[[VAL_19]] {uniq_name = "_QMmFaddressing_explicit_shapeEx"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
! CHECK:             %[[VAL_21:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_22:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_23:.*]] = hlfir.designate %[[VAL_20]]#0 (%[[VAL_21]], %[[VAL_22]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
! CHECK:             fir.call @_QPtakes_scalar(%[[VAL_23]]) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accVar(%[[VAL_17]] : !fir.box<!fir.array<?x?xf32>>) to var(%[[VAL_16]]#0 : !fir.box<!fir.array<?x?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPaddressing_assumed_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFaddressing_assumed_shapeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFaddressing_assumed_shapeEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           %[[VAL_3:.*]] = acc.copyin var(%[[VAL_2]]#0 : !fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_3]] : !fir.box<!fir.array<?x?xf32>>) {
! CHECK:             %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]] dummy_scope %[[VAL_4]] skip_rebox {uniq_name = "_QMmFaddressing_assumed_shapeEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:             %[[VAL_6:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_7:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_6]], %[[VAL_7]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
! CHECK:             fir.call @_QPtakes_scalar(%[[VAL_8]]) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accVar(%[[VAL_3]] : !fir.box<!fir.array<?x?xf32>>) to var(%[[VAL_2]]#0 : !fir.box<!fir.array<?x?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPaddressing_contiguous_assumed_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x", fir.contiguous},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFaddressing_contiguous_assumed_shapeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[ARG0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape_shift %[[VAL_5]], %[[VAL_4]]#1, %[[VAL_8]], %[[VAL_7]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_9]]) dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QMmFaddressing_contiguous_assumed_shapeEx"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
! CHECK:           %[[VAL_11:.*]] = acc.copyin var(%[[VAL_10]]#0 : !fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_11]] : !fir.box<!fir.array<?x?xf32>>) {
! CHECK:             %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:             %[[VAL_13:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_9]]) dummy_scope %[[VAL_13]] {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QMmFaddressing_contiguous_assumed_shapeEx"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
! CHECK:             %[[VAL_15:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_16:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_14]]#0 (%[[VAL_15]], %[[VAL_16]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
! CHECK:             fir.call @_QPtakes_scalar(%[[VAL_17]]) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accVar(%[[VAL_11]] : !fir.box<!fir.array<?x?xf32>>) to var(%[[VAL_10]]#0 : !fir.box<!fir.array<?x?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPaddressing_pointer(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmFaddressing_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_2:.*]] = acc.copyin varPtr(%[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) {
! CHECK:             %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmFaddressing_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
! CHECK:             %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:             %[[VAL_6:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_7:.*]] = arith.constant 3 : index
! CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]] (%[[VAL_6]], %[[VAL_7]])  : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index) -> !fir.ref<f32>
! CHECK:             fir.call @_QPtakes_scalar(%[[VAL_8]]) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) to varPtr(%[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_optional_scalar(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMmFtest_optional_scalarEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_2:.*]] = acc.copyin varPtr(%[[VAL_1]]#0 : !fir.ref<f32>) -> !fir.ref<f32> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_2]] : !fir.ref<f32>) {
! CHECK:             %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMmFtest_optional_scalarEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             %[[VAL_5:.*]] = fir.is_present %[[VAL_4]]#0 : (!fir.ref<f32>) -> i1
! CHECK:             %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (!fir.ref<f32>) {
! CHECK:               fir.result %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:             } else {
! CHECK:               %[[VAL_7:.*]] = fir.absent !fir.ref<f32>
! CHECK:               fir.result %[[VAL_7]] : !fir.ref<f32>
! CHECK:             }
! CHECK:             fir.call @_QPtakes_optional_scalar(%[[VAL_6]]) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_2]] : !fir.ref<f32>) to varPtr(%[[VAL_1]]#0 : !fir.ref<f32>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_optional_explicit_cst_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_2]]) dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMmFtest_optional_explicit_cst_shapeEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:           %[[VAL_4:.*]] = acc.copyin varPtr(%[[VAL_3]]#0 : !fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_4]] : !fir.ref<!fir.array<100xf32>>) {
! CHECK:             %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_2]]) dummy_scope %[[VAL_5]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMmFtest_optional_explicit_cst_shapeEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:             %[[VAL_7:.*]] = fir.is_present %[[VAL_6]]#0 : (!fir.ref<!fir.array<100xf32>>) -> i1
! CHECK:             %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.ref<!fir.array<100xf32>>) {
! CHECK:               fir.result %[[VAL_6]]#0 : !fir.ref<!fir.array<100xf32>>
! CHECK:             } else {
! CHECK:               %[[VAL_9:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
! CHECK:               fir.result %[[VAL_9]] : !fir.ref<!fir.array<100xf32>>
! CHECK:             }
! CHECK:             fir.call @_QPtakes_optional_explicit_cst_shape(%[[VAL_8]]) fastmath<contract> : (!fir.ref<!fir.array<100xf32>>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_4]] : !fir.ref<!fir.array<100xf32>>) to varPtr(%[[VAL_3]]#0 : !fir.ref<!fir.array<100xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_optional_explicit_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QMmFtest_optional_explicit_shapeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_8]]) dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMmFtest_optional_explicit_shapeEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           %[[VAL_10:.*]] = acc.copyin varPtr(%[[VAL_9]]#1 : !fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_10]] : !fir.ref<!fir.array<?xf32>>) {
! CHECK:             %[[VAL_11:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_8]]) dummy_scope %[[VAL_11]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMmFtest_optional_explicit_shapeEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:             %[[VAL_13:.*]] = fir.is_present %[[VAL_12]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] -> (!fir.ref<!fir.array<?xf32>>) {
! CHECK:               fir.result %[[VAL_12]]#1 : !fir.ref<!fir.array<?xf32>>
! CHECK:             } else {
! CHECK:               %[[VAL_15:.*]] = fir.absent !fir.ref<!fir.array<?xf32>>
! CHECK:               fir.result %[[VAL_15]] : !fir.ref<!fir.array<?xf32>>
! CHECK:             }
! CHECK:             fir.call @_QPtakes_optional_explicit_shape(%[[VAL_14]], %[[VAL_1]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>, !fir.ref<i32>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_10]] : !fir.ref<!fir.array<?xf32>>) to varPtr(%[[VAL_9]]#1 : !fir.ref<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_optional_assumed_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMmFtest_optional_assumed_shapeEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_2:.*]] = acc.copyin var(%[[VAL_1]]#0 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_2]] : !fir.box<!fir.array<?xf32>>) {
! CHECK:             %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] skip_rebox {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMmFtest_optional_assumed_shapeEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:             %[[VAL_5:.*]] = fir.is_present %[[VAL_4]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:             %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (!fir.box<!fir.array<?xf32>>) {
! CHECK:               fir.result %[[VAL_4]]#0 : !fir.box<!fir.array<?xf32>>
! CHECK:             } else {
! CHECK:               %[[VAL_7:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:               fir.result %[[VAL_7]] : !fir.box<!fir.array<?xf32>>
! CHECK:             }
! CHECK:             fir.call @_QPtakes_optional_assumed_shape(%[[VAL_6]]) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accVar(%[[VAL_2]] : !fir.box<!fir.array<?xf32>>) to var(%[[VAL_1]]#0 : !fir.box<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QMmPtest_optional_pointer(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<optional, pointer>, uniq_name = "_QMmFtest_optional_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_2:.*]] = acc.copyin varPtr(%[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           acc.parallel dataOperands(%[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
! CHECK:             %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<optional, pointer>, uniq_name = "_QMmFtest_optional_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:             fir.call @_QPtakes_optional_pointer(%[[VAL_4]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> ()
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           acc.copyout accPtr(%[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) to varPtr(%[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {dataClause = #acc<data_clause acc_copy>, name = "x"}
! CHECK:           return
! CHECK:         }
