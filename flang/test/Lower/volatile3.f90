! RUN: bbc %s -o - | FileCheck %s

! Test that all combinations of volatile pointer and target are properly lowered -
! note that a volatile pointer implies that the target is volatile, even if not specified

program p
   integer, volatile              :: volatile_integer, volatile_array(10), &
                                     volatile_array_2d(10,10)
   integer                        :: nonvolatile_array(10)
   integer, volatile, target      :: volatile_integer_target, volatile_array_target(10)
   integer, target                :: nonvolatile_integer_target, nonvolatile_array_target(10)
   integer, volatile, &
            pointer, dimension(:) :: volatile_array_pointer
   integer, pointer, dimension(:) :: nonvolatile_array_pointer

   volatile_array_pointer    => volatile_array_target
   volatile_array_pointer    => nonvolatile_array_target
   nonvolatile_array_pointer => volatile_array_target
   nonvolatile_array_pointer => nonvolatile_array_target

   call sub_nonvolatile_array(volatile_array)
   call sub_volatile_array_assumed_shape(volatile_array)
   call sub_volatile_array(volatile_array)

   call sub_volatile_array_assumed_shape(nonvolatile_array)
   call sub_volatile_array(nonvolatile_array)

   call sub_volatile_array_pointer(volatile_array_pointer)
   call sub_volatile_array_pointer(nonvolatile_array_pointer)

   call sub_volatile_array_assumed_shape(volatile_array(1:10:1))
   call sub_volatile_array_assumed_shape_2d(volatile_array_2d(1:10:1,:))
contains
   subroutine sub_nonvolatile_array(arr)
      integer :: arr(10)
      arr(1) = 5
   end subroutine
   subroutine sub_volatile_array_assumed_shape(arr)
      integer, volatile, dimension(:) :: arr
      arr(1) = 5
   end subroutine
   subroutine sub_volatile_array_assumed_shape_2d(arr)
      integer, volatile, dimension(:,:) :: arr
      arr(1,1) = 5
   end subroutine
   subroutine sub_volatile_array(arr)
      integer, volatile, dimension(10) :: arr
      arr(1) = 5
   end subroutine
   subroutine sub_volatile_array_pointer(arr)
      integer, volatile, dimension(:), pointer :: arr
      arr(1) = 5
   end subroutine
end program

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFEnonvolatile_array) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_3]]) {uniq_name = "_QFEnonvolatile_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QFEnonvolatile_array_pointer) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEnonvolatile_array_pointer"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QFEnonvolatile_array_target) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_3]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEnonvolatile_array_target"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QFEnonvolatile_integer_target) : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEnonvolatile_integer_target"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QFEvolatile_array) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_12:.*]] = fir.volatile_cast %[[VAL_11]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_3]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_array"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QFEvolatile_array_2d) : !fir.ref<!fir.array<10x10xi32>>
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_1]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_16:.*]] = fir.volatile_cast %[[VAL_14]] : (!fir.ref<!fir.array<10x10xi32>>) -> !fir.ref<!fir.array<10x10xi32>, volatile>
! CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]](%[[VAL_15]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_array_2d"} : (!fir.ref<!fir.array<10x10xi32>, volatile>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x10xi32>, volatile>, !fir.ref<!fir.array<10x10xi32>, volatile>)
! CHECK:           %[[VAL_18:.*]] = fir.address_of(@_QFEvolatile_array_pointer) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_19:.*]] = fir.volatile_cast %[[VAL_18]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEvolatile_array_pointer"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[VAL_21:.*]] = fir.address_of(@_QFEvolatile_array_target) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_22:.*]] = fir.volatile_cast %[[VAL_21]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_22]](%[[VAL_3]]) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEvolatile_array_target"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_24:.*]] = fir.alloca i32 {bindc_name = "volatile_integer", uniq_name = "_QFEvolatile_integer"}
! CHECK:           %[[VAL_25:.*]] = fir.volatile_cast %[[VAL_24]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_25]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_integer"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[VAL_27:.*]] = fir.address_of(@_QFEvolatile_integer_target) : !fir.ref<i32>
! CHECK:           %[[VAL_28:.*]] = fir.volatile_cast %[[VAL_27]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_28]] {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEvolatile_integer_target"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[VAL_30:.*]] = fir.embox %[[VAL_23]]#0(%[[VAL_3]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_30]] to %[[VAL_20]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_31:.*]] = fir.embox %[[VAL_8]]#0(%[[VAL_3]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_32:.*]] = fir.volatile_cast %[[VAL_31]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_32]] to %[[VAL_20]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_33:.*]] = fir.volatile_cast %[[VAL_30]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_33]] to %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           fir.store %[[VAL_31]] to %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_34:.*]] = fir.volatile_cast %[[VAL_13]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           fir.call @_QFPsub_nonvolatile_array(%[[VAL_34]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[VAL_35:.*]] = fir.embox %[[VAL_13]]#0(%[[VAL_3]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_36:.*]] = fir.volatile_cast %[[VAL_35]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[VAL_37]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           fir.call @_QFPsub_volatile_array(%[[VAL_34]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[VAL_38:.*]] = fir.embox %[[VAL_4]]#0(%[[VAL_3]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[VAL_39]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           fir.call @_QFPsub_volatile_array(%[[VAL_4]]#0) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_20]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
! CHECK:           fir.call @_QFPsub_volatile_array_pointer(%[[VAL_40]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
! CHECK:           %[[VAL_41:.*]] = fir.volatile_cast %[[VAL_6]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
! CHECK:           fir.call @_QFPsub_volatile_array_pointer(%[[VAL_41]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
! CHECK:           %[[VAL_42:.*]] = hlfir.designate %[[VAL_13]]#0 (%[[VAL_0]]:%[[VAL_1]]:%[[VAL_0]])  shape %[[VAL_3]] : (!fir.ref<!fir.array<10xi32>, volatile>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_43:.*]] = fir.embox %[[VAL_42]](%[[VAL_3]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_44:.*]] = fir.volatile_cast %[[VAL_43]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[VAL_45]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           %[[VAL_46:.*]] = hlfir.designate %[[VAL_17]]#0 (%[[VAL_0]]:%[[VAL_1]]:%[[VAL_0]], %[[VAL_0]]:%[[VAL_1]]:%[[VAL_0]])  shape %[[VAL_15]] : (!fir.ref<!fir.array<10x10xi32>, volatile>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>, volatile>
! CHECK:           %[[VAL_47:.*]] = fir.volatile_cast %[[VAL_46]] : (!fir.box<!fir.array<10x10xi32>, volatile>) -> !fir.box<!fir.array<10x10xi32>>
! CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape_2d(%[[VAL_48]]) fastmath<contract> : (!fir.box<!fir.array<?x?xi32>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_nonvolatile_array(
! CHECK-SAME:                                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 5 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) dummy_scope %[[VAL_4]] {uniq_name = "_QFFsub_nonvolatile_arrayEarr"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_7:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_1]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_volatile_array_assumed_shape(
! CHECK-SAME:                                                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 5 : i32
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>, volatile>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_volatile_array_assumed_shapeEarr"} : (!fir.box<!fir.array<?xi32>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>, volatile>, !fir.box<!fir.array<?xi32>, volatile>)
! CHECK:           %[[VAL_6:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_1]])  : (!fir.box<!fir.array<?xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_6]] : i32, !fir.ref<i32, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_volatile_array_assumed_shape_2d(
! CHECK-SAME:                                                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 5 : i32
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>, volatile>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_volatile_array_assumed_shape_2dEarr"} : (!fir.box<!fir.array<?x?xi32>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>, volatile>, !fir.box<!fir.array<?x?xi32>, volatile>)
! CHECK:           %[[VAL_6:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_1]], %[[VAL_1]])  : (!fir.box<!fir.array<?x?xi32>, volatile>, index, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_6]] : i32, !fir.ref<i32, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_volatile_array(
! CHECK-SAME:                                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 5 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_5]]) dummy_scope %[[VAL_4]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_volatile_arrayEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_8:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_1]])  : (!fir.ref<!fir.array<10xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_8]] : i32, !fir.ref<i32, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_volatile_array_pointer(
! CHECK-SAME:                                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 5 : i32
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFFsub_volatile_array_pointerEarr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_7:.*]] = hlfir.designate %[[VAL_6]] (%[[VAL_1]])  : (!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.ref<i32, volatile>
! CHECK:           return
! CHECK:         }
