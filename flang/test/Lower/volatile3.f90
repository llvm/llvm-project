! RUN: bbc %s -o - | FileCheck %s

! Test that all combinations of volatile pointer and target are properly lowered -
! note that a volatile pointer implies that the target is volatile, even if not specified

program p
   integer, volatile              :: volatile_integer, volatile_array(10), &
                                     volatile_array_2d(10,10)
   integer, volatile, pointer      :: volatile_integer_pointer
   integer                        :: nonvolatile_array(10)
   integer, volatile, target      :: volatile_integer_target, volatile_array_target(10)
   integer, target                :: nonvolatile_integer_target, nonvolatile_array_target(10)
   integer, volatile, &
            pointer, dimension(:) :: volatile_array_pointer
   integer, pointer, dimension(:) :: nonvolatile_array_pointer

   volatile_array_pointer    => volatile_array_target
   volatile_array_pointer    => nonvolatile_array_target
   volatile_array_pointer    => null(volatile_array_pointer)
   nonvolatile_array_pointer => volatile_array_target
   nonvolatile_array_pointer => nonvolatile_array_target
   volatile_integer_pointer  => volatile_integer_target
   volatile_integer_pointer  => null(volatile_integer_pointer)

   call sub_nonvolatile_array(volatile_array)
   call sub_volatile_array_assumed_shape(volatile_array)
   call sub_volatile_array(volatile_array)

   call sub_volatile_array_assumed_shape(nonvolatile_array)
   call sub_volatile_array(nonvolatile_array)

   call sub_volatile_array_pointer(volatile_array_pointer)
   call sub_volatile_array_pointer(nonvolatile_array_pointer)

   call sub_volatile_array_assumed_shape(volatile_array(1:10:1))
   call sub_volatile_array_assumed_shape_2d(volatile_array_2d(1:10:1,:))

   call sub_select_rank(volatile_array)
   call sub_select_rank(volatile_array_2d)
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
   subroutine sub_select_rank(arr)
      integer, volatile :: arr(..)
      select rank(arr)
      rank(1)
         arr(1) = 5
      rank(4)
         arr(1,1,1,1) = 5
      end select
   end subroutine
end program


! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<i32>, volatile>
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QFEnonvolatile_array) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) {uniq_name = "_QFEnonvolatile_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QFEnonvolatile_array_pointer) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEnonvolatile_array_pointer"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QFEnonvolatile_array_target) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_6]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEnonvolatile_array_target"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_12:.*]] = fir.address_of(@_QFEnonvolatile_integer_target) : !fir.ref<i32>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEnonvolatile_integer_target"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QFEvolatile_array) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_15:.*]] = fir.volatile_cast %[[VAL_14]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]](%[[VAL_6]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_array"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_17:.*]] = fir.address_of(@_QFEvolatile_array_2d) : !fir.ref<!fir.array<10x10xi32>>
! CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_19:.*]] = fir.volatile_cast %[[VAL_17]] : (!fir.ref<!fir.array<10x10xi32>>) -> !fir.ref<!fir.array<10x10xi32>, volatile>
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]](%[[VAL_18]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_array_2d"} : (!fir.ref<!fir.array<10x10xi32>, volatile>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x10xi32>, volatile>, !fir.ref<!fir.array<10x10xi32>, volatile>)
! CHECK:           %[[VAL_21:.*]] = fir.address_of(@_QFEvolatile_array_pointer) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_22:.*]] = fir.volatile_cast %[[VAL_21]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_22]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEvolatile_array_pointer"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[VAL_24:.*]] = fir.address_of(@_QFEvolatile_array_target) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_25:.*]] = fir.volatile_cast %[[VAL_24]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_25]](%[[VAL_6]]) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEvolatile_array_target"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_27:.*]] = fir.alloca i32 {bindc_name = "volatile_integer", uniq_name = "_QFEvolatile_integer"}
! CHECK:           %[[VAL_28:.*]] = fir.volatile_cast %[[VAL_27]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_28]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_integer"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[VAL_30:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "volatile_integer_pointer", uniq_name = "_QFEvolatile_integer_pointer"}
! CHECK:           %[[VAL_31:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK:           %[[VAL_32:.*]] = fir.embox %[[VAL_31]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           fir.store %[[VAL_32]] to %[[VAL_30]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_33:.*]] = fir.volatile_cast %[[VAL_30]] : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>
! CHECK:           %[[VAL_34:.*]]:2 = hlfir.declare %[[VAL_33]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEvolatile_integer_pointer"} : (!fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>)
! CHECK:           %[[VAL_35:.*]] = fir.address_of(@_QFEvolatile_integer_target) : !fir.ref<i32>
! CHECK:           %[[VAL_36:.*]] = fir.volatile_cast %[[VAL_35]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_37:.*]]:2 = hlfir.declare %[[VAL_36]] {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEvolatile_integer_target"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[VAL_38:.*]] = fir.embox %[[VAL_26]]#0(%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_38]] to %[[VAL_23]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_39:.*]] = fir.embox %[[VAL_11]]#0(%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_40:.*]] = fir.volatile_cast %[[VAL_39]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_40]] to %[[VAL_23]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_41:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_42:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_43:.*]] = fir.embox %[[VAL_41]](%[[VAL_42]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_43]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>
! CHECK:           %[[VAL_44:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>)
! CHECK:           %[[VAL_45:.*]] = fir.load %[[VAL_44]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>
! CHECK:           %[[VAL_46:.*]]:3 = fir.box_dims %[[VAL_45]], %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, index) -> (index, index, index)
! CHECK:           %[[VAL_47:.*]] = fir.shift %[[VAL_46]]#0 : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_48:.*]] = fir.rebox %[[VAL_45]](%[[VAL_47]]) : (!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_48]] to %[[VAL_23]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_49:.*]] = fir.volatile_cast %[[VAL_38]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_49]] to %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           fir.store %[[VAL_39]] to %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_50:.*]] = fir.embox %[[VAL_37]]#0 : (!fir.ref<i32, volatile>) -> !fir.box<!fir.ptr<i32>, volatile>
! CHECK:           fir.store %[[VAL_50]] to %[[VAL_34]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>
! CHECK:           %[[VAL_51:.*]] = fir.embox %[[VAL_31]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>, volatile>
! CHECK:           fir.store %[[VAL_51]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<i32>, volatile>>
! CHECK:           %[[VAL_52:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.box<!fir.ptr<i32>, volatile>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>, volatile>>, !fir.ref<!fir.box<!fir.ptr<i32>, volatile>>)
! CHECK:           %[[VAL_53:.*]] = fir.load %[[VAL_52]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>, volatile>>
! CHECK:           %[[VAL_54:.*]] = fir.box_addr %[[VAL_53]] : (!fir.box<!fir.ptr<i32>, volatile>) -> !fir.ptr<i32>
! CHECK:           %[[VAL_55:.*]] = fir.embox %[[VAL_54]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           %[[VAL_56:.*]] = fir.volatile_cast %[[VAL_55]] : (!fir.box<!fir.ptr<i32>>) -> !fir.box<!fir.ptr<i32>, volatile>
! CHECK:           fir.store %[[VAL_56]] to %[[VAL_34]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>
! CHECK:           %[[VAL_57:.*]] = fir.volatile_cast %[[VAL_16]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           fir.call @_QFPsub_nonvolatile_array(%[[VAL_57]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[VAL_58:.*]] = fir.embox %[[VAL_16]]#0(%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_59:.*]] = fir.volatile_cast %[[VAL_58]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[VAL_60]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           fir.call @_QFPsub_volatile_array(%[[VAL_57]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[VAL_61:.*]] = fir.embox %[[VAL_7]]#0(%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_62:.*]] = fir.convert %[[VAL_61]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[VAL_62]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           fir.call @_QFPsub_volatile_array(%[[VAL_7]]#0) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_23]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
! CHECK:           fir.call @_QFPsub_volatile_array_pointer(%[[VAL_63]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
! CHECK:           %[[VAL_64:.*]] = fir.volatile_cast %[[VAL_9]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
! CHECK:           fir.call @_QFPsub_volatile_array_pointer(%[[VAL_64]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
! CHECK:           %[[VAL_65:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_0]]:%[[VAL_2]]:%[[VAL_0]])  shape %[[VAL_6]] : (!fir.ref<!fir.array<10xi32>, volatile>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_66:.*]] = fir.embox %[[VAL_65]](%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_67:.*]] = fir.volatile_cast %[[VAL_66]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[VAL_68]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           %[[VAL_69:.*]] = hlfir.designate %[[VAL_20]]#0 (%[[VAL_0]]:%[[VAL_2]]:%[[VAL_0]], %[[VAL_0]]:%[[VAL_2]]:%[[VAL_0]])  shape %[[VAL_18]] : (!fir.ref<!fir.array<10x10xi32>, volatile>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>, volatile>
! CHECK:           %[[VAL_70:.*]] = fir.volatile_cast %[[VAL_69]] : (!fir.box<!fir.array<10x10xi32>, volatile>) -> !fir.box<!fir.array<10x10xi32>>
! CHECK:           %[[VAL_71:.*]] = fir.convert %[[VAL_70]] : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape_2d(%[[VAL_71]]) fastmath<contract> : (!fir.box<!fir.array<?x?xi32>>) -> ()
! CHECK:           %[[VAL_72:.*]] = fir.convert %[[VAL_59]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QFPsub_select_rank(%[[VAL_72]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()
! CHECK:           %[[VAL_73:.*]] = fir.embox %[[VAL_20]]#0(%[[VAL_18]]) : (!fir.ref<!fir.array<10x10xi32>, volatile>, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>, volatile>
! CHECK:           %[[VAL_74:.*]] = fir.volatile_cast %[[VAL_73]] : (!fir.box<!fir.array<10x10xi32>, volatile>) -> !fir.box<!fir.array<10x10xi32>>
! CHECK:           %[[VAL_75:.*]] = fir.convert %[[VAL_74]] : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QFPsub_select_rank(%[[VAL_75]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()
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

! CHECK-LABEL:   func.func private @_QFPsub_select_rank(
! CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:i32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 5 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 4 : i8
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.box<!fir.array<*:i32>>) -> !fir.box<!fir.array<*:i32>, volatile>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] dummy_scope %[[VAL_5]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_select_rankEarr"} : (!fir.box<!fir.array<*:i32>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<*:i32>, volatile>, !fir.box<!fir.array<*:i32>, volatile>)
! CHECK:           %[[VAL_8:.*]] = fir.volatile_cast %[[VAL_7]]#0 : (!fir.box<!fir.array<*:i32>, volatile>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<*:i32>>) -> !fir.box<none>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAIsAssumedSize(%[[VAL_9]]) : (!fir.box<none>) -> i1
! CHECK:           cf.cond_br %[[VAL_10]], ^bb4, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_11:.*]] = fir.box_rank %[[VAL_7]]#0 : (!fir.box<!fir.array<*:i32>, volatile>) -> i8
! CHECK:           fir.select_case %[[VAL_11]] : i8 [#fir.point, %[[VAL_4]], ^bb2, #fir.point, %[[VAL_3]], ^bb3, unit, ^bb4]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.box<!fir.array<*:i32>, volatile>) -> !fir.box<!fir.array<?xi32>, volatile>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_select_rankEarr"} : (!fir.box<!fir.array<?xi32>, volatile>) -> (!fir.box<!fir.array<?xi32>, volatile>, !fir.box<!fir.array<?xi32>, volatile>)
! CHECK:           %[[VAL_14:.*]] = hlfir.designate %[[VAL_13]]#0 (%[[VAL_1]])  : (!fir.box<!fir.array<?xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_14]] : i32, !fir.ref<i32, volatile>
! CHECK:           cf.br ^bb4
! CHECK:         ^bb3:
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.box<!fir.array<*:i32>, volatile>) -> !fir.box<!fir.array<?x?x?x?xi32>, volatile>
! CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_select_rankEarr"} : (!fir.box<!fir.array<?x?x?x?xi32>, volatile>) -> (!fir.box<!fir.array<?x?x?x?xi32>, volatile>, !fir.box<!fir.array<?x?x?x?xi32>, volatile>)
! CHECK:           %[[VAL_17:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_1]], %[[VAL_1]], %[[VAL_1]], %[[VAL_1]])  : (!fir.box<!fir.array<?x?x?x?xi32>, volatile>, index, index, index, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_17]] : i32, !fir.ref<i32, volatile>
! CHECK:           cf.br ^bb4
! CHECK:         ^bb4:
! CHECK:           return
! CHECK:         }
