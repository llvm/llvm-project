! Test conversion of MutableBoxValue to value.
! RUN: bbc -emit-hlfir -polymorphic-type -I nowhere %s -o - | FileCheck %s

subroutine test_int_allocatable(a)
  integer, allocatable :: a
  print *, a
end subroutine test_int_allocatable
! CHECK-LABEL:   func.func @_QPtest_int_allocatable(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "a"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_int_allocatableEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           %[[VAL_2:.*]] = arith.constant -1 : i32
! CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_5:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_6:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_4]], %[[VAL_5]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.heap<i32>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_6]], %[[VAL_9]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
! CHECK:           %[[VAL_11:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_6]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

subroutine test_int_pointer(p)
  integer, pointer :: p
  print *, p
end subroutine test_int_pointer
! CHECK-LABEL:   func.func @_QPtest_int_pointer(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "p"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_int_pointerEp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           %[[VAL_2:.*]] = arith.constant -1 : i32
! CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_5:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_6:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_4]], %[[VAL_5]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ptr<i32>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_6]], %[[VAL_9]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
! CHECK:           %[[VAL_11:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_6]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

subroutine test_char_allocatable(a)
  character(11), allocatable :: a
  integer :: i
  i = len_trim(a)
end subroutine test_char_allocatable
! CHECK-LABEL:   func.func @_QPtest_char_allocatable(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>> {fir.bindc_name = "a"}) {
! CHECK:           %[[VAL_1:.*]] = arith.constant 11 : index
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_char_allocatableEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_char_allocatableEi"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFtest_char_allocatableEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>>
! CHECK:           %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.heap<!fir.char<1,11>>>) -> !fir.heap<!fir.char<1,11>>
! CHECK:           %[[VAL_1B:.*]] = arith.constant 11 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]] = arith.constant true
! CHECK:           %[[VAL_11:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_12:.*]] = arith.subi %[[VAL_1B]], %[[VAL_7]] : index
! CHECK:           %[[VAL_13:.*]]:2 = fir.iterate_while (%[[VAL_14:.*]] = %[[VAL_12]] to %[[VAL_9]] step %[[VAL_8]]) and (%[[VAL_15:.*]] = %[[VAL_10]]) iter_args(%[[VAL_16:.*]] = %[[VAL_12]]) -> (index) {
! CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.heap<!fir.char<1,11>>) -> !fir.ref<!fir.array<11x!fir.char<1>>>
! CHECK:             %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_14]] : (!fir.ref<!fir.array<11x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i8>
! CHECK:             %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_20]] : i8
! CHECK:             fir.result %[[VAL_21]], %[[VAL_14]] : i1, index
! CHECK:           }
! CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_23:.*]]#1, %[[VAL_7]] : index
! CHECK:           %[[VAL_24:.*]] = arith.select %[[VAL_23]]#0, %[[VAL_9]], %[[VAL_22]] : index
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
! CHECK:           hlfir.assign %[[VAL_25]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

subroutine test_char_pointer(p)
  character(11), pointer :: p
  integer :: i
  i = len_trim(p)
end subroutine test_char_pointer
! CHECK-LABEL:   func.func @_QPtest_char_pointer(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,11>>>> {fir.bindc_name = "p"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_char_pointerEi"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_char_pointerEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 11 : index
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_3]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_char_pointerEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,11>>>>, index) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,11>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,11>>>>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,11>>>>
! CHECK:           %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.ptr<!fir.char<1,11>>>) -> !fir.ptr<!fir.char<1,11>>
! CHECK:           %[[VAL_3B:.*]] = arith.constant 11 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]] = arith.constant true
! CHECK:           %[[VAL_11:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_12:.*]] = arith.subi %[[VAL_3B]], %[[VAL_7]] : index
! CHECK:           %[[VAL_13:.*]]:2 = fir.iterate_while (%[[VAL_14:.*]] = %[[VAL_12]] to %[[VAL_9]] step %[[VAL_8]]) and (%[[VAL_15:.*]] = %[[VAL_10]]) iter_args(%[[VAL_16:.*]] = %[[VAL_12]]) -> (index) {
! CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.ptr<!fir.char<1,11>>) -> !fir.ref<!fir.array<11x!fir.char<1>>>
! CHECK:             %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_14]] : (!fir.ref<!fir.array<11x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i8>
! CHECK:             %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_20]] : i8
! CHECK:             fir.result %[[VAL_21]], %[[VAL_14]] : i1, index
! CHECK:           }
! CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_23:.*]]#1, %[[VAL_7]] : index
! CHECK:           %[[VAL_24:.*]] = arith.select %[[VAL_23]]#0, %[[VAL_9]], %[[VAL_22]] : index
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
! CHECK:           hlfir.assign %[[VAL_25]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

subroutine test_dyn_char_allocatable(a)
  character(*), allocatable :: a
  integer :: i
  i = len_trim(a)
end subroutine test_dyn_char_allocatable
! CHECK-LABEL:   func.func @_QPtest_dyn_char_allocatable(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>> {fir.bindc_name = "a"}) {
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_dyn_char_allocatableEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dyn_char_allocatableEi"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest_dyn_char_allocatableEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]] = arith.constant true
! CHECK:           %[[VAL_12:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_2]], %[[VAL_8]] : index
! CHECK:           %[[VAL_14:.*]]:2 = fir.iterate_while (%[[VAL_15:.*]] = %[[VAL_13]] to %[[VAL_10]] step %[[VAL_9]]) and (%[[VAL_16:.*]] = %[[VAL_11]]) iter_args(%[[VAL_17:.*]] = %[[VAL_13]]) -> (index) {
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_18]], %[[VAL_15]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i8>
! CHECK:             %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_21]] : i8
! CHECK:             fir.result %[[VAL_22]], %[[VAL_15]] : i1, index
! CHECK:           }
! CHECK:           %[[VAL_23:.*]] = arith.addi %[[VAL_24:.*]]#1, %[[VAL_8]] : index
! CHECK:           %[[VAL_25:.*]] = arith.select %[[VAL_24]]#0, %[[VAL_10]], %[[VAL_23]] : index
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (index) -> i32
! CHECK:           hlfir.assign %[[VAL_26]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

subroutine test_dyn_char_pointer(p)
  character(*), pointer :: p
  integer :: i
  i = len_trim(p)
end subroutine test_dyn_char_pointer
! CHECK-LABEL:   func.func @_QPtest_dyn_char_pointer(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "p"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dyn_char_pointerEi"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_dyn_char_pointerEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_elesize %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_4]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_dyn_char_pointerEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, index) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]] = arith.constant true
! CHECK:           %[[VAL_12:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_4]], %[[VAL_8]] : index
! CHECK:           %[[VAL_14:.*]]:2 = fir.iterate_while (%[[VAL_15:.*]] = %[[VAL_13]] to %[[VAL_10]] step %[[VAL_9]]) and (%[[VAL_16:.*]] = %[[VAL_11]]) iter_args(%[[VAL_17:.*]] = %[[VAL_13]]) -> (index) {
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_7]] : (!fir.ptr<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_18]], %[[VAL_15]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i8>
! CHECK:             %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_21]] : i8
! CHECK:             fir.result %[[VAL_22]], %[[VAL_15]] : i1, index
! CHECK:           }
! CHECK:           %[[VAL_23:.*]] = arith.addi %[[VAL_24:.*]]#1, %[[VAL_8]] : index
! CHECK:           %[[VAL_25:.*]] = arith.select %[[VAL_24]]#0, %[[VAL_10]], %[[VAL_23]] : index
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (index) -> i32
! CHECK:           hlfir.assign %[[VAL_26]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

subroutine test_derived_allocatable(l)
  type t
  end type t
  type(t), allocatable :: a1
  class(t), allocatable :: a2, r
  logical :: l
  r = merge(a1, a2, l)
end subroutine test_derived_allocatable
! CHECK-LABEL:   func.func @_QPtest_derived_allocatable(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>> {bindc_name = "a1", uniq_name = "_QFtest_derived_allocatableEa1"}
! CHECK:           %[[VAL_2:.*]] = fir.zero_bits !fir.heap<!fir.type<_QFtest_derived_allocatableTt>>
! CHECK:           %[[VAL_3:.*]] = fir.embox %[[VAL_2]] : (!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>) -> !fir.box<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_derived_allocatableEa1"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>> {bindc_name = "a2", uniq_name = "_QFtest_derived_allocatableEa2"}
! CHECK:           %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.type<_QFtest_derived_allocatableTt>>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]] : (!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>) -> !fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_derived_allocatableEa2"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>)
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_derived_allocatableEl"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>> {bindc_name = "r", uniq_name = "_QFtest_derived_allocatableEr"}
! CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.type<_QFtest_derived_allocatableTt>>
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]] : (!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>) -> !fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>
! CHECK:           fir.store %[[VAL_12]] to %[[VAL_10]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_10]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_derived_allocatableEr"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>)
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>
! CHECK:           %[[VAL_14B:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>) -> !fir.heap<!fir.type<_QFtest_derived_allocatableTt>>
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.logical<4>) -> i1
! CHECK:           %[[VAL_18:.*]] = fir.box_addr %[[VAL_15]] : (!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>) -> !fir.heap<!fir.type<_QFtest_derived_allocatableTt>>
! CHECK:           %[[VAL_19:.*]] = arith.select %[[VAL_17]], %[[VAL_14B]], %[[VAL_18]] : !fir.heap<!fir.type<_QFtest_derived_allocatableTt>>
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>) -> (!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>, !fir.heap<!fir.type<_QFtest_derived_allocatableTt>>)
! CHECK:           %[[VAL_21:.*]] = hlfir.as_expr %[[VAL_20]]#0 : (!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>) -> !hlfir.expr<!fir.type<_QFtest_derived_allocatableTt>>
! CHECK:           hlfir.assign %[[VAL_21]] to %[[VAL_13]]#0 realloc : !hlfir.expr<!fir.type<_QFtest_derived_allocatableTt>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_allocatableTt>>>>
! CHECK:           hlfir.destroy %[[VAL_21]] : !hlfir.expr<!fir.type<_QFtest_derived_allocatableTt>>
! CHECK:           return
! CHECK:         }

subroutine test_derived_pointer(l)
  type t
  end type t
  type(t), pointer :: a1
  class(t), allocatable :: a2, r
  logical :: l
  r = merge(a1, a2, l)
end subroutine test_derived_pointer
! CHECK-LABEL:   func.func @_QPtest_derived_pointer(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>> {bindc_name = "a1", uniq_name = "_QFtest_derived_pointerEa1"}
! CHECK:           %[[VAL_2:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QFtest_derived_pointerTt>>
! CHECK:           %[[VAL_3:.*]] = fir.embox %[[VAL_2]] : (!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>) -> !fir.box<!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>>>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_derived_pointerEa1"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>>>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>> {bindc_name = "a2", uniq_name = "_QFtest_derived_pointerEa2"}
! CHECK:           %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.type<_QFtest_derived_pointerTt>>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]] : (!fir.heap<!fir.type<_QFtest_derived_pointerTt>>) -> !fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_derived_pointerEa2"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>)
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_derived_pointerEl"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>> {bindc_name = "r", uniq_name = "_QFtest_derived_pointerEr"}
! CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.type<_QFtest_derived_pointerTt>>
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]] : (!fir.heap<!fir.type<_QFtest_derived_pointerTt>>) -> !fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>
! CHECK:           fir.store %[[VAL_12]] to %[[VAL_10]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_10]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_derived_pointerEr"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>)
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>>>
! CHECK:           %[[VAL_14B:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>>) -> !fir.ptr<!fir.type<_QFtest_derived_pointerTt>>
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.logical<4>) -> i1
! CHECK:           %[[VAL_18:.*]] = fir.box_addr %[[VAL_15]] : (!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>) -> !fir.ptr<!fir.type<_QFtest_derived_pointerTt>>
! CHECK:           %[[VAL_19:.*]] = arith.select %[[VAL_17]], %[[VAL_14B]], %[[VAL_18]] : !fir.ptr<!fir.type<_QFtest_derived_pointerTt>>
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>) -> (!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>, !fir.ptr<!fir.type<_QFtest_derived_pointerTt>>)
! CHECK:           %[[VAL_21:.*]] = hlfir.as_expr %[[VAL_20]]#0 : (!fir.ptr<!fir.type<_QFtest_derived_pointerTt>>) -> !hlfir.expr<!fir.type<_QFtest_derived_pointerTt>>
! CHECK:           hlfir.assign %[[VAL_21]] to %[[VAL_13]]#0 realloc : !hlfir.expr<!fir.type<_QFtest_derived_pointerTt>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtest_derived_pointerTt>>>>
! CHECK:           hlfir.destroy %[[VAL_21]] : !hlfir.expr<!fir.type<_QFtest_derived_pointerTt>>
! CHECK:           return
! CHECK:         }
