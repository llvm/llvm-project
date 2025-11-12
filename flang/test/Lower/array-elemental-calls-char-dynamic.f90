! Test lowering of elemental calls to character function where the
! result length is not a compile time constant.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! The vector subscript must not be read when computing the result length
! before the elemental loop because the argument array could be zero sized.
subroutine test_vector_subscripted_arg(c, vector_subscript)
  interface
    elemental function bug_145151_1(c_dummy)
      character(*), intent(in) :: c_dummy
      character(len(c_dummy, KIND=8)) :: bug_145151_1
    end
  end interface
  integer(8) :: vector_subscript(:)
  character(*) :: c(:)
  c = bug_145151_1(c(vector_subscript))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_vector_subscripted_arg(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?xi64>> {fir.bindc_name = "vector_subscript"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_vector_subscripted_argEc"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_vector_subscripted_argEvector_subscript"} : (!fir.box<!fir.array<?xi64>>, !fir.dscope) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
! CHECK:           %[[VAL_3:.*]] = fir.box_elesize %[[VAL_1]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]]#0, %[[VAL_4]] : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_3]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_vector_subscripted_argFbug_145151_1Ec_dummy"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : index
! CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_11]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = hlfir.elemental %[[VAL_6]] typeparams %[[VAL_14]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[VAL_16:.*]]: index):
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_16]])  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
! CHECK:             %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<i64>
! CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_1]]#0 (%[[VAL_18]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i64, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_20:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_14]] : index) {bindc_name = ".result"}
! CHECK:             %[[VAL_21:.*]] = fir.call @_QPbug_145151_1(%[[VAL_20]], %[[VAL_14]], %[[VAL_19]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<!fir.char<1,?>>, index, !fir.boxchar<1>) -> !fir.boxchar<1>
! CHECK:             %[[VAL_22:.*]]:2 = hlfir.declare %[[VAL_20]] typeparams %[[VAL_14]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:             %[[VAL_23:.*]] = arith.constant false
! CHECK:             %[[VAL_24:.*]] = hlfir.as_expr %[[VAL_22]]#0 move %[[VAL_23]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             hlfir.yield_element %[[VAL_24]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_15]] to %[[VAL_1]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           hlfir.destroy %[[VAL_15]] : !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   fir.global @_QMm_bug_145151_2Ei : i64 {
! CHECK:           %[[VAL_0:.*]] = fir.zero_bits i64
! CHECK:           fir.has_value %[[VAL_0]] : i64
! CHECK:         }




module m_bug_145151_2
  integer(8) :: i
end module

! Test that module variables used in the result specification expressions
! are mapped correctly.
subroutine test_module_variable(c, x)
  interface
    elemental function bug_145151_2(x)
      use m_bug_145151_2, only : i
      real, value :: x
      character(i) :: bug_145151_2
    end
  end interface
  character(*) :: c(:)
  real :: x(:)
  c = bug_145151_2(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_module_variable(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_module_variableEc"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_module_variableEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]]#0, %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QMm_bug_145151_2Ei) : !fir.ref<i64>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QMm_bug_145151_2Ei"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_10]] : index
! CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_9]], %[[VAL_10]] : index
! CHECK:           %[[VAL_13:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_12]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[VAL_14:.*]]: index):
! CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_14]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<f32>
! CHECK:             %[[VAL_17:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_12]] : index) {bindc_name = ".result"}
! CHECK:             %[[VAL_18:.*]] = fir.call @_QPbug_145151_2(%[[VAL_17]], %[[VAL_12]], %[[VAL_16]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<!fir.char<1,?>>, index, f32) -> !fir.boxchar<1>
! CHECK:             %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_17]] typeparams %[[VAL_12]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:             %[[VAL_20:.*]] = arith.constant false
! CHECK:             %[[VAL_21:.*]] = hlfir.as_expr %[[VAL_19]]#0 move %[[VAL_20]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             hlfir.yield_element %[[VAL_21]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_1]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           hlfir.destroy %[[VAL_13]] : !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }


! Test that optional arguments are not dereferenced unconditionally when preparing
! them for inquiries inside the result specification expressions.
subroutine test_present(res, x, opt)
  interface
    elemental function f_opt(x, opt)
      real, intent(in)  :: x
      real, intent(in), optional :: opt
      character(merge(10,20, present(opt))) :: f_opt
    end
  end interface
  character(*) :: res(:)
  real :: x(:)
  real, optional :: opt(:)
  res = f_opt(x, opt)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_present(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "res"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "opt", fir.optional}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_presentEopt"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_presentEres"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_presentEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_4:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_4]] -> (!fir.ref<f32>) {
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> !fir.ref<f32>
! CHECK:             fir.result %[[VAL_10]] : !fir.ref<f32>
! CHECK:           } else {
! CHECK:             %[[VAL_11:.*]] = fir.absent !fir.ref<f32>
! CHECK:             fir.result %[[VAL_11]] : !fir.ref<f32>
! CHECK:           }
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_8]] {fortran_attrs = #fir.var_attrs<intent_in, optional>, uniq_name = "_QFtest_presentFf_optEopt"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_13:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_14:.*]] = arith.constant 20 : i32
! CHECK:           %[[VAL_15:.*]] = fir.is_present %[[VAL_12]]#0 : (!fir.ref<f32>) -> i1
! CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_13]], %[[VAL_14]] : i32
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_19]] : index
! CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_18]], %[[VAL_19]] : index
! CHECK:           %[[VAL_22:.*]] = hlfir.elemental %[[VAL_7]] typeparams %[[VAL_21]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[VAL_23:.*]]: index):
! CHECK:             %[[VAL_24:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_23]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[VAL_25:.*]] = fir.if %[[VAL_4]] -> (!fir.ref<f32>) {
! CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_1]]#0 (%[[VAL_23]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:               fir.result %[[VAL_26]] : !fir.ref<f32>
! CHECK:             } else {
! CHECK:               %[[VAL_27:.*]] = fir.absent !fir.ref<f32>
! CHECK:               fir.result %[[VAL_27]] : !fir.ref<f32>
! CHECK:             }
! CHECK:             %[[VAL_28:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_21]] : index) {bindc_name = ".result"}
! CHECK:             %[[VAL_29:.*]] = fir.call @_QPf_opt(%[[VAL_28]], %[[VAL_21]], %[[VAL_24]], %[[VAL_25]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<!fir.char<1,?>>, index, !fir.ref<f32>, !fir.ref<f32>) -> !fir.boxchar<1>
! CHECK:             %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_28]] typeparams %[[VAL_21]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:             %[[VAL_31:.*]] = arith.constant false
! CHECK:             %[[VAL_32:.*]] = hlfir.as_expr %[[VAL_30]]#0 move %[[VAL_31]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             hlfir.yield_element %[[VAL_32]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_22]] to %[[VAL_2]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           hlfir.destroy %[[VAL_22]] : !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }

! Test that inquiries about the dynamic type of arguments are handled inside the
! elemental result specification expressions.
subroutine test_polymorphic(res, p1, p2)
  type t
  end type
  interface
    elemental function f_poly(p1, p2)
      import :: t
      class(t), intent(in)  :: p1, p2
      character(merge(10,20, STORAGE_SIZE(p1).lt.STORAGE_SIZE(p2))) :: f_poly
    end
  end interface
  character(*) :: res(:)
  class(t), intent(in)  :: p1(:), p2(:)
  res = f_poly(p1, p2)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_polymorphic(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "res"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>> {fir.bindc_name = "p1"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>> {fir.bindc_name = "p2"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_polymorphicEp1"} : (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>, !fir.dscope) -> (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>, !fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_polymorphicEp2"} : (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>, !fir.dscope) -> (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>, !fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_polymorphicEres"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_1]]#0, %[[VAL_4]] : (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> !fir.ref<!fir.type<_QFtest_polymorphicTt>>
! CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_8]] source_box %[[VAL_1]]#0 : (!fir.ref<!fir.type<_QFtest_polymorphicTt>>, !fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>) -> !fir.class<!fir.type<_QFtest_polymorphicTt>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> !fir.ref<!fir.type<_QFtest_polymorphicTt>>
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]] source_box %[[VAL_2]]#0 : (!fir.ref<!fir.type<_QFtest_polymorphicTt>>, !fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>) -> !fir.class<!fir.type<_QFtest_polymorphicTt>>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_polymorphicFf_polyEp1"} : (!fir.class<!fir.type<_QFtest_polymorphicTt>>) -> (!fir.class<!fir.type<_QFtest_polymorphicTt>>, !fir.class<!fir.type<_QFtest_polymorphicTt>>)
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_polymorphicFf_polyEp2"} : (!fir.class<!fir.type<_QFtest_polymorphicTt>>) -> (!fir.class<!fir.type<_QFtest_polymorphicTt>>, !fir.class<!fir.type<_QFtest_polymorphicTt>>)
! CHECK:           %[[VAL_15:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_16:.*]] = arith.constant 20 : i32
! CHECK:           %[[VAL_17:.*]] = fir.box_elesize %[[VAL_13]]#1 : (!fir.class<!fir.type<_QFtest_polymorphicTt>>) -> i32
! CHECK:           %[[VAL_18:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_19:.*]] = arith.muli %[[VAL_17]], %[[VAL_18]] : i32
! CHECK:           %[[VAL_20:.*]] = fir.box_elesize %[[VAL_14]]#1 : (!fir.class<!fir.type<_QFtest_polymorphicTt>>) -> i32
! CHECK:           %[[VAL_21:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_22:.*]] = arith.muli %[[VAL_20]], %[[VAL_21]] : i32
! CHECK:           %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_22]] : i32
! CHECK:           %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[VAL_15]], %[[VAL_16]] : i32
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:           %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_27]] : index
! CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_26]], %[[VAL_27]] : index
! CHECK:           %[[VAL_30:.*]] = hlfir.elemental %[[VAL_6]] typeparams %[[VAL_29]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[VAL_31:.*]]: index):
! CHECK:             %[[VAL_32:.*]] = hlfir.designate %[[VAL_1]]#0 (%[[VAL_31]])  : (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>, index) -> !fir.class<!fir.type<_QFtest_polymorphicTt>>
! CHECK:             %[[VAL_33:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_31]])  : (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphicTt>>>, index) -> !fir.class<!fir.type<_QFtest_polymorphicTt>>
! CHECK:             %[[VAL_34:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_29]] : index) {bindc_name = ".result"}
! CHECK:             %[[VAL_35:.*]] = fir.call @_QPf_poly(%[[VAL_34]], %[[VAL_29]], %[[VAL_32]], %[[VAL_33]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<!fir.char<1,?>>, index, !fir.class<!fir.type<_QFtest_polymorphicTt>>, !fir.class<!fir.type<_QFtest_polymorphicTt>>) -> !fir.boxchar<1>
! CHECK:             %[[VAL_36:.*]]:2 = hlfir.declare %[[VAL_34]] typeparams %[[VAL_29]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:             %[[VAL_37:.*]] = arith.constant false
! CHECK:             %[[VAL_38:.*]] = hlfir.as_expr %[[VAL_36]]#0 move %[[VAL_37]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             hlfir.yield_element %[[VAL_38]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_30]] to %[[VAL_3]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           hlfir.destroy %[[VAL_30]] : !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }

! Test that no copy of VALUE argument is made before the loop when
! evaluating the result specification expression (while a copy
! of the argument elements have to be made inside the loop).
subroutine test_value(c)
  interface
    elemental function f_value(c_dummy)
      character(*), value :: c_dummy
      character(len(c_dummy, KIND=8)) :: f_value
    end
  end interface
  character(*) :: c(:)
  c = f_value(c)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_value(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_valueEc"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_1]]#0, %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_7:.*]] = fir.box_elesize %[[VAL_1]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]] typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFtest_valueFf_valueEc_dummy"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (index) -> i64
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_11]] : index
! CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_10]], %[[VAL_11]] : index
! CHECK:           %[[VAL_14:.*]] = hlfir.elemental %[[VAL_4]] typeparams %[[VAL_13]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[VAL_15:.*]]: index):
! CHECK:             %[[VAL_16:.*]] = fir.box_elesize %[[VAL_1]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_1]]#0 (%[[VAL_15]])  typeparams %[[VAL_16]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_18:.*]] = hlfir.as_expr %[[VAL_17]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             %[[VAL_19:.*]]:3 = hlfir.associate %[[VAL_18]] typeparams %[[VAL_16]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:             %[[VAL_20:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_13]] : index) {bindc_name = ".result"}
! CHECK:             %[[VAL_21:.*]] = fir.call @_QPf_value(%[[VAL_20]], %[[VAL_13]], %[[VAL_19]]#0) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<!fir.char<1,?>>, index, !fir.boxchar<1>) -> !fir.boxchar<1>
! CHECK:             %[[VAL_22:.*]]:2 = hlfir.declare %[[VAL_20]] typeparams %[[VAL_13]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:             %[[VAL_23:.*]] = arith.constant false
! CHECK:             %[[VAL_24:.*]] = hlfir.as_expr %[[VAL_22]]#0 move %[[VAL_23]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:             hlfir.end_associate %[[VAL_19]]#1, %[[VAL_19]]#2 : !fir.ref<!fir.char<1,?>>, i1
! CHECK:             hlfir.yield_element %[[VAL_24]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_1]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           hlfir.destroy %[[VAL_14]] : !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }
