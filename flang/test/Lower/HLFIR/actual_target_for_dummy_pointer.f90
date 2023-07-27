! Test actual TARGET argument association to dummy POINTER:
! RUN: bbc -emit-hlfir --polymorphic-type -o - -I nowhere %s 2>&1 | FileCheck %s

module target_to_pointer_types
  type t1
  end type t1
end module target_to_pointer_types

subroutine integer_scalar()
  interface
     subroutine integer_scalar_callee(p)
       integer, pointer, intent(in) :: p
     end subroutine integer_scalar_callee
     subroutine integer_scalar_uclass_callee(p)
       class(*), pointer, intent(in) :: p
     end subroutine integer_scalar_uclass_callee
  end interface
  integer, target :: i
  call integer_scalar_callee(i)
  call integer_scalar_uclass_callee(i)
end subroutine integer_scalar
! CHECK-LABEL:   func.func @_QPinteger_scalar() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.ptr<none>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "i", fir.target, uniq_name = "_QFinteger_scalarEi"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFinteger_scalarEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           fir.call @_QPinteger_scalar_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> ()
! CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<i32>) -> !fir.class<!fir.ptr<none>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK:           fir.call @_QPinteger_scalar_uclass_callee(%[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine integer_assumed_shape_array(i)
  interface
     subroutine integer_assumed_shape_array_callee(p)
       integer, pointer, intent(in) :: p(:)
     end subroutine integer_assumed_shape_array_callee
     subroutine integer_assumed_shape_array_uclass_callee(p)
       class(*), pointer, intent(in) :: p(:)
     end subroutine integer_assumed_shape_array_uclass_callee
  end interface
  integer, target :: i(:)
  call integer_assumed_shape_array_callee(i)
  call integer_assumed_shape_array_uclass_callee(i)
end subroutine integer_assumed_shape_array
! CHECK-LABEL:   func.func @_QPinteger_assumed_shape_array(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFinteger_assumed_shape_arrayEi"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_4:.*]] = fir.rebox %[[VAL_3]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           fir.call @_QPinteger_assumed_shape_array_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> ()
! CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_3]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPinteger_assumed_shape_array_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine integer_explicit_shape_array()
  interface
     subroutine integer_explicit_shape_array_callee(p)
       integer, pointer, intent(in) :: p(:)
     end subroutine integer_explicit_shape_array_callee
     subroutine integer_explicit_shape_array_uclass_callee(p)
       class(*), pointer, intent(in) :: p(:)
     end subroutine integer_explicit_shape_array_uclass_callee
  end interface
  integer, target :: i(100)
  call integer_explicit_shape_array_callee(i)
  call integer_explicit_shape_array_uclass_callee(i)
end subroutine integer_explicit_shape_array
! CHECK-LABEL:   func.func @_QPinteger_explicit_shape_array() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_2:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "i", fir.target, uniq_name = "_QFinteger_explicit_shape_arrayEi"}
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFinteger_explicit_shape_arrayEi"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_5]]#1(%[[VAL_6]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           fir.call @_QPinteger_explicit_shape_array_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> ()
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_5]]#1(%[[VAL_8]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPinteger_explicit_shape_array_uclass_callee(%[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine char_scalar()
  interface
     subroutine char_scalar_explicit_len_callee(p)
       character(2), pointer, intent(in) :: p
     end subroutine char_scalar_explicit_len_callee
     subroutine char_scalar_assumed_len_callee(p)
       character(*), pointer, intent(in) :: p
     end subroutine char_scalar_assumed_len_callee
     subroutine char_scalar_uclass_callee(p)
       class(*), pointer, intent(in) :: p
     end subroutine char_scalar_uclass_callee
  end interface
  character(2), target :: a
  call char_scalar_explicit_len_callee(a)
  call char_scalar_assumed_len_callee(a)
  call char_scalar_uclass_callee(a)
end subroutine char_scalar
! CHECK-LABEL:   func.func @_QPchar_scalar() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.ptr<none>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,2>>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.char<1,2> {bindc_name = "a", fir.target, uniq_name = "_QFchar_scalarEa"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] typeparams %[[VAL_3]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFchar_scalarEa"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,2>>) -> !fir.box<!fir.ptr<!fir.char<1,2>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,2>>>>
! CHECK:           fir.call @_QPchar_scalar_explicit_len_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,2>>>>) -> ()
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_7]] typeparams %[[VAL_3]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           fir.call @_QPchar_scalar_assumed_len_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> ()
! CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,2>>) -> !fir.class<!fir.ptr<none>>
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK:           fir.call @_QPchar_scalar_uclass_callee(%[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine char_assumed_shape_array(a1, a2)
  interface
     subroutine char_assumed_shape_array_explicit_len_callee(p)
       character(2), pointer, intent(in) :: p(:)
     end subroutine char_assumed_shape_array_explicit_len_callee
     subroutine char_assumed_shape_array_assumed_len_callee(p)
       character(*), pointer, intent(in) :: p(:)
     end subroutine char_assumed_shape_array_assumed_len_callee
     subroutine char_assumed_shape_array_uclass_callee(p)
       class(*), pointer, intent(in) :: p(:)
     end subroutine char_assumed_shape_array_uclass_callee
  end interface
  character(2), target :: a1(:)
  character(*), target :: a2(:)
  call char_assumed_shape_array_explicit_len_callee(a1)
  call char_assumed_shape_array_assumed_len_callee(a1)
  call char_assumed_shape_array_uclass_callee(a1)
  call char_assumed_shape_array_explicit_len_callee(a2)
  call char_assumed_shape_array_assumed_len_callee(a2)
  call char_assumed_shape_array_uclass_callee(a2)
end subroutine char_assumed_shape_array
! CHECK-LABEL:   func.func @_QPchar_assumed_shape_array(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,2>>> {fir.bindc_name = "a1", fir.target},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a2", fir.target}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_8]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFchar_assumed_shape_arrayEa1"} : (!fir.box<!fir.array<?x!fir.char<1,2>>>, index) -> (!fir.box<!fir.array<?x!fir.char<1,2>>>, !fir.box<!fir.array<?x!fir.char<1,2>>>)
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFchar_assumed_shape_arrayEa2"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_11:.*]] = fir.rebox %[[VAL_9]]#1 : (!fir.box<!fir.array<?x!fir.char<1,2>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>
! CHECK:           fir.store %[[VAL_11]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           fir.call @_QPchar_assumed_shape_array_explicit_len_callee(%[[VAL_7]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>>) -> ()
! CHECK:           %[[VAL_12:.*]] = fir.rebox %[[VAL_9]]#1 : (!fir.box<!fir.array<?x!fir.char<1,2>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           fir.store %[[VAL_12]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           fir.call @_QPchar_assumed_shape_array_assumed_len_callee(%[[VAL_6]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()
! CHECK:           %[[VAL_13:.*]] = fir.rebox %[[VAL_9]]#1 : (!fir.box<!fir.array<?x!fir.char<1,2>>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPchar_assumed_shape_array_uclass_callee(%[[VAL_5]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           %[[VAL_14:.*]] = fir.rebox %[[VAL_10]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           fir.call @_QPchar_assumed_shape_array_explicit_len_callee(%[[VAL_4]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>>) -> ()
! CHECK:           %[[VAL_15:.*]] = fir.rebox %[[VAL_10]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           fir.call @_QPchar_assumed_shape_array_assumed_len_callee(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()
! CHECK:           %[[VAL_16:.*]] = fir.rebox %[[VAL_10]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_2]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPchar_assumed_shape_array_uclass_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine char_explicit_shape_array(a2)
  interface
     subroutine char_explicit_shape_array_explicit_len_callee(p)
       character(2), pointer, intent(in) :: p(:)
     end subroutine char_explicit_shape_array_explicit_len_callee
     subroutine char_explicit_shape_array_assumed_len_callee(p)
       character(*), pointer, intent(in) :: p(:)
     end subroutine char_explicit_shape_array_assumed_len_callee
     subroutine char_explicit_shape_array_uclass_callee(p)
       class(*), pointer, intent(in) :: p(:)
     end subroutine char_explicit_shape_array_uclass_callee
  end interface
  character(2), target :: a1(100)
  character(*), target :: a2(100)
  call char_explicit_shape_array_explicit_len_callee(a1)
  call char_explicit_shape_array_assumed_len_callee(a1)
  call char_explicit_shape_array_uclass_callee(a1)
  call char_explicit_shape_array_explicit_len_callee(a2)
  call char_explicit_shape_array_assumed_len_callee(a2)
  call char_explicit_shape_array_uclass_callee(a2)
end subroutine char_explicit_shape_array
! CHECK-LABEL:   func.func @_QPchar_explicit_shape_array(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "a2", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>
! CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.array<100x!fir.char<1,2>> {bindc_name = "a1", fir.target, uniq_name = "_QFchar_explicit_shape_arrayEa1"}
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFchar_explicit_shape_arrayEa1"} : (!fir.ref<!fir.array<100x!fir.char<1,2>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,2>>>, !fir.ref<!fir.array<100x!fir.char<1,2>>>)
! CHECK:           %[[VAL_12:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
! CHECK:           %[[VAL_14:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_15]]) typeparams %[[VAL_12]]#1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFchar_explicit_shape_arrayEa2"} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
! CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_11]]#1 : (!fir.ref<!fir.array<100x!fir.char<1,2>>>) -> !fir.ref<!fir.array<?x!fir.char<1,2>>>
! CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_17]]) : (!fir.ref<!fir.array<?x!fir.char<1,2>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           fir.call @_QPchar_explicit_shape_array_explicit_len_callee(%[[VAL_6]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>>) -> ()
! CHECK:           %[[VAL_20:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_11]]#1 : (!fir.ref<!fir.array<100x!fir.char<1,2>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_21]](%[[VAL_20]]) typeparams %[[VAL_7]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           fir.call @_QPchar_explicit_shape_array_assumed_len_callee(%[[VAL_5]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()
! CHECK:           %[[VAL_23:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_24:.*]] = fir.embox %[[VAL_11]]#1(%[[VAL_23]]) : (!fir.ref<!fir.array<100x!fir.char<1,2>>>, !fir.shape<1>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_24]] to %[[VAL_4]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPchar_explicit_shape_array_uclass_callee(%[[VAL_4]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           %[[VAL_25:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_16]]#1 : (!fir.ref<!fir.array<100x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,2>>>
! CHECK:           %[[VAL_27:.*]] = fir.embox %[[VAL_26]](%[[VAL_25]]) : (!fir.ref<!fir.array<?x!fir.char<1,2>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>
! CHECK:           fir.store %[[VAL_27]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           fir.call @_QPchar_explicit_shape_array_explicit_len_callee(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,2>>>>>) -> ()
! CHECK:           %[[VAL_28:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_16]]#1 : (!fir.ref<!fir.array<100x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:           %[[VAL_30:.*]] = fir.embox %[[VAL_29]](%[[VAL_28]]) typeparams %[[VAL_12]]#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           fir.store %[[VAL_30]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           fir.call @_QPchar_explicit_shape_array_assumed_len_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()
! CHECK:           %[[VAL_31:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_32:.*]] = fir.embox %[[VAL_16]]#1(%[[VAL_31]]) : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_32]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPchar_explicit_shape_array_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine type_scalar()
  use target_to_pointer_types
  interface
     subroutine type_scalar_callee(p)
       use target_to_pointer_types
       type(t1), pointer, intent(in) :: p
     end subroutine type_scalar_callee
     subroutine type_scalar_class_callee(p)
       use target_to_pointer_types
       class(t1), pointer, intent(in) :: p
     end subroutine type_scalar_class_callee
     subroutine type_scalar_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p
     end subroutine type_scalar_uclass_callee
  end interface
  type(t1), target :: t
  call type_scalar_callee(t)
  call type_scalar_class_callee(t)
  call type_scalar_uclass_callee(t)
end subroutine type_scalar
! CHECK-LABEL:   func.func @_QPtype_scalar() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.ptr<none>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.type<_QMtarget_to_pointer_typesTt1> {bindc_name = "t", fir.target, uniq_name = "_QFtype_scalarEt"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtype_scalarEt"} : (!fir.ref<!fir.type<_QMtarget_to_pointer_typesTt1>>) -> (!fir.ref<!fir.type<_QMtarget_to_pointer_typesTt1>>, !fir.ref<!fir.type<_QMtarget_to_pointer_typesTt1>>)
! CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_4]]#1 : (!fir.ref<!fir.type<_QMtarget_to_pointer_typesTt1>>) -> !fir.box<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.call @_QPtype_scalar_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>>) -> ()
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_4]]#1 : (!fir.ref<!fir.type<_QMtarget_to_pointer_typesTt1>>) -> !fir.class<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.call @_QPtype_scalar_class_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>>) -> ()
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_4]]#1 : (!fir.ref<!fir.type<_QMtarget_to_pointer_typesTt1>>) -> !fir.class<!fir.ptr<none>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK:           fir.call @_QPtype_scalar_uclass_callee(%[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine type_assumed_shape_array(t)
  use target_to_pointer_types
  interface
     subroutine type_assumed_shape_array_callee(p)
       use target_to_pointer_types
       type(t1), pointer, intent(in) :: p(:)
     end subroutine type_assumed_shape_array_callee
     subroutine type_assumed_shape_array_class_callee(p)
       use target_to_pointer_types
       class(t1), pointer, intent(in) :: p(:)
     end subroutine type_assumed_shape_array_class_callee
     subroutine type_assumed_shape_array_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p(:)
     end subroutine type_assumed_shape_array_uclass_callee
  end interface
  type(t1), target :: t(:)
  call type_assumed_shape_array_callee(t)
  call type_assumed_shape_array_class_callee(t)
  call type_assumed_shape_array_uclass_callee(t)
end subroutine type_assumed_shape_array
! CHECK-LABEL:   func.func @_QPtype_assumed_shape_array(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>> {fir.bindc_name = "t", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtype_assumed_shape_arrayEt"} : (!fir.box<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>, !fir.box<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>)
! CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.box<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>
! CHECK:           fir.call @_QPtype_assumed_shape_array_callee(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>) -> ()
! CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.box<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>
! CHECK:           fir.call @_QPtype_assumed_shape_array_class_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>) -> ()
! CHECK:           %[[VAL_7:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.box<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPtype_assumed_shape_array_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine type_explicit_shape_array()
  use target_to_pointer_types
  interface
     subroutine type_explicit_shape_array_callee(p)
       use target_to_pointer_types
       type(t1), pointer, intent(in) :: p(:)
     end subroutine type_explicit_shape_array_callee
     subroutine type_explicit_shape_array_class_callee(p)
       use target_to_pointer_types
       class(t1), pointer, intent(in) :: p(:)
     end subroutine type_explicit_shape_array_class_callee
     subroutine type_explicit_shape_array_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p(:)
     end subroutine type_explicit_shape_array_uclass_callee
  end interface
  type(t1), target :: t(100)
  call type_explicit_shape_array_callee(t)
  call type_explicit_shape_array_class_callee(t)
  call type_explicit_shape_array_uclass_callee(t)
end subroutine type_explicit_shape_array
! CHECK-LABEL:   func.func @_QPtype_explicit_shape_array() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>> {bindc_name = "t", fir.target, uniq_name = "_QFtype_explicit_shape_arrayEt"}
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_5]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtype_explicit_shape_arrayEt"} : (!fir.ref<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>, !fir.ref<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>)
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_6]]#1(%[[VAL_7]]) : (!fir.ref<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>
! CHECK:           fir.call @_QPtype_explicit_shape_array_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>) -> ()
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]] = fir.embox %[[VAL_6]]#1(%[[VAL_9]]) : (!fir.ref<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>, !fir.shape<1>) -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>
! CHECK:           fir.call @_QPtype_explicit_shape_array_class_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>) -> ()
! CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_6]]#1(%[[VAL_11]]) : (!fir.ref<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>, !fir.shape<1>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_12]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPtype_explicit_shape_array_uclass_callee(%[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine class_scalar(t)
  use target_to_pointer_types
  interface
     subroutine class_scalar_callee(p)
       use target_to_pointer_types
       type(t1), pointer, intent(in) :: p
     end subroutine class_scalar_callee
     subroutine class_scalar_class_callee(p)
       use target_to_pointer_types
       class(t1), pointer, intent(in) :: p
     end subroutine class_scalar_class_callee
     subroutine class_scalar_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p
     end subroutine class_scalar_uclass_callee
  end interface
  class(t1), target :: t
  call class_scalar_callee(t)
  call class_scalar_class_callee(t)
  call class_scalar_uclass_callee(t)
end subroutine class_scalar
! CHECK-LABEL:   func.func @_QPclass_scalar(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.class<!fir.type<_QMtarget_to_pointer_typesTt1>> {fir.bindc_name = "t", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<none>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFclass_scalarEt"} : (!fir.class<!fir.type<_QMtarget_to_pointer_typesTt1>>) -> (!fir.class<!fir.type<_QMtarget_to_pointer_typesTt1>>, !fir.class<!fir.type<_QMtarget_to_pointer_typesTt1>>)
! CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.type<_QMtarget_to_pointer_typesTt1>>) -> !fir.box<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.call @_QPclass_scalar_callee(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>>) -> ()
! CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.type<_QMtarget_to_pointer_typesTt1>>) -> !fir.class<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.call @_QPclass_scalar_class_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMtarget_to_pointer_typesTt1>>>>) -> ()
! CHECK:           %[[VAL_7:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.type<_QMtarget_to_pointer_typesTt1>>) -> !fir.class<!fir.ptr<none>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK:           fir.call @_QPclass_scalar_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine class_assumed_shape_array(t)
  use target_to_pointer_types
  interface
     subroutine class_assumed_shape_array_callee(p)
       use target_to_pointer_types
       type(t1), pointer, intent(in) :: p(:)
     end subroutine class_assumed_shape_array_callee
     subroutine class_assumed_shape_array_class_callee(p)
       use target_to_pointer_types
       class(t1), pointer, intent(in) :: p(:)
     end subroutine class_assumed_shape_array_class_callee
     subroutine class_assumed_shape_array_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p(:)
     end subroutine class_assumed_shape_array_uclass_callee
  end interface
  class(t1), target :: t(:)
  call class_assumed_shape_array_callee(t)
  call class_assumed_shape_array_class_callee(t)
  call class_assumed_shape_array_uclass_callee(t)
end subroutine class_assumed_shape_array
! CHECK-LABEL:   func.func @_QPclass_assumed_shape_array(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>> {fir.bindc_name = "t", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFclass_assumed_shape_arrayEt"} : (!fir.class<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> (!fir.class<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>, !fir.class<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>)
! CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>
! CHECK:           fir.call @_QPclass_assumed_shape_array_callee(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>) -> ()
! CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>
! CHECK:           fir.call @_QPclass_assumed_shape_array_class_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>) -> ()
! CHECK:           %[[VAL_7:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPclass_assumed_shape_array_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine class_explicit_shape_array(t)
  use target_to_pointer_types
  interface
     subroutine class_explicit_shape_array_callee(p)
       use target_to_pointer_types
       type(t1), pointer, intent(in) :: p(:)
     end subroutine class_explicit_shape_array_callee
     subroutine class_explicit_shape_array_class_callee(p)
       use target_to_pointer_types
       class(t1), pointer, intent(in) :: p(:)
     end subroutine class_explicit_shape_array_class_callee
     subroutine class_explicit_shape_array_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p(:)
     end subroutine class_explicit_shape_array_uclass_callee
  end interface
  class(t1), target :: t(100)
  call class_explicit_shape_array_callee(t)
  call class_explicit_shape_array_class_callee(t)
  call class_explicit_shape_array_uclass_callee(t)
end subroutine class_explicit_shape_array
! CHECK-LABEL:   func.func @_QPclass_explicit_shape_array(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.class<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>> {fir.bindc_name = "t", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFclass_explicit_shape_arrayEt"} : (!fir.class<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> (!fir.class<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>, !fir.class<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>)
! CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>
! CHECK:           fir.call @_QPclass_explicit_shape_array_callee(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>) -> ()
! CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>
! CHECK:           fir.call @_QPclass_explicit_shape_array_class_callee(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMtarget_to_pointer_typesTt1>>>>>) -> ()
! CHECK:           %[[VAL_7:.*]] = fir.rebox %[[VAL_4]]#1 : (!fir.class<!fir.array<100x!fir.type<_QMtarget_to_pointer_typesTt1>>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPclass_explicit_shape_array_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine uclass_scalar(t)
  use target_to_pointer_types
  interface
     subroutine uclass_scalar_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p
     end subroutine uclass_scalar_uclass_callee
  end interface
  class(*), target :: t
  call uclass_scalar_uclass_callee(t)
end subroutine uclass_scalar
! CHECK-LABEL:   func.func @_QPuclass_scalar(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.class<none> {fir.bindc_name = "t", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<none>>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFuclass_scalarEt"} : (!fir.class<none>) -> (!fir.class<none>, !fir.class<none>)
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]]#1 : (!fir.class<none>) -> !fir.class<!fir.ptr<none>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK:           fir.call @_QPuclass_scalar_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine uclass_assumed_shape_array(t)
  use target_to_pointer_types
  interface
     subroutine uclass_assumed_shape_array_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p(:)
     end subroutine uclass_assumed_shape_array_uclass_callee
  end interface
  class(*), target :: t(:)
  call uclass_assumed_shape_array_uclass_callee(t)
end subroutine uclass_assumed_shape_array
! CHECK-LABEL:   func.func @_QPuclass_assumed_shape_array(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.class<!fir.array<?xnone>> {fir.bindc_name = "t", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFuclass_assumed_shape_arrayEt"} : (!fir.class<!fir.array<?xnone>>) -> (!fir.class<!fir.array<?xnone>>, !fir.class<!fir.array<?xnone>>)
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]]#1 : (!fir.class<!fir.array<?xnone>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPuclass_assumed_shape_array_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine uclass_explicit_shape_array(t)
  use target_to_pointer_types
  interface
     subroutine uclass_explicit_shape_array_uclass_callee(p)
       use target_to_pointer_types
       class(*), pointer, intent(in) :: p(:)
     end subroutine uclass_explicit_shape_array_uclass_callee
  end interface
  class(*), target :: t(100)
  call uclass_explicit_shape_array_uclass_callee(t)
end subroutine uclass_explicit_shape_array
! CHECK-LABEL:   func.func @_QPuclass_explicit_shape_array(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.class<!fir.array<100xnone>> {fir.bindc_name = "t", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFuclass_explicit_shape_arrayEt"} : (!fir.class<!fir.array<100xnone>>) -> (!fir.class<!fir.array<100xnone>>, !fir.class<!fir.array<100xnone>>)
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]]#1 : (!fir.class<!fir.array<100xnone>>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPuclass_explicit_shape_array_uclass_callee(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }
