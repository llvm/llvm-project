! Test passing character array to unlimited polymorphic array pointer.
! Regression test from https://github.com/llvm/llvm-project/issues/150749

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine char_explicit_shape_array(a2)
interface
subroutine char_explicit_shape_array_uclass_callee(p)
 class(*), pointer, intent(in) :: p(:)
end subroutine char_explicit_shape_array_uclass_callee
end interface
character(*), target :: a2(100)
call char_explicit_shape_array_uclass_callee(a2)
end subroutine char_explicit_shape_array
! CHECK-LABEL:   func.func @_QPchar_explicit_shape_array(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "a2", fir.target}) {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_5]]) typeparams %[[VAL_2]]#1 dummy_scope %[[VAL_1]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFchar_explicit_shape_arrayEa2"} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_6]]#1(%[[VAL_7]]) typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:           fir.call @_QPchar_explicit_shape_array_uclass_callee(%[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> ()
! CHECK:           return
! CHECK:         }
