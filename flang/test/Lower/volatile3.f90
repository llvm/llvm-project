! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s

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

! CHECK-LABEL:   func.func @_QQmain() {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>, volatile>
! CHECK:           %[[ALLOCA_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QFEnonvolatile_array) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ADDRESS_OF_0]](%[[SHAPE_0]]) {uniq_name = "_QFEnonvolatile_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[ADDRESS_OF_1:.*]] = fir.address_of(@_QFEnonvolatile_array_pointer) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ADDRESS_OF_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEnonvolatile_array_pointer"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:           %[[ADDRESS_OF_2:.*]] = fir.address_of(@_QFEnonvolatile_array_target) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[ADDRESS_OF_2]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEnonvolatile_array_target"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[ADDRESS_OF_3:.*]] = fir.address_of(@_QFEnonvolatile_integer_target) : !fir.ref<i32>
! CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[ADDRESS_OF_3]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEnonvolatile_integer_target"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ADDRESS_OF_4:.*]] = fir.address_of(@_QFEvolatile_array) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ADDRESS_OF_4]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[DECLARE_4:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_array"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[ADDRESS_OF_5:.*]] = fir.address_of(@_QFEvolatile_array_2d) : !fir.ref<!fir.array<10x10xi32>>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_2]], %[[CONSTANT_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VOLATILE_CAST_1:.*]] = fir.volatile_cast %[[ADDRESS_OF_5]] : (!fir.ref<!fir.array<10x10xi32>>) -> !fir.ref<!fir.array<10x10xi32>, volatile>
! CHECK:           %[[DECLARE_5:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_1]](%[[SHAPE_1]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_array_2d"} : (!fir.ref<!fir.array<10x10xi32>, volatile>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x10xi32>, volatile>, !fir.ref<!fir.array<10x10xi32>, volatile>)
! CHECK:           %[[ADDRESS_OF_6:.*]] = fir.address_of(@_QFEvolatile_array_pointer) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VOLATILE_CAST_2:.*]] = fir.volatile_cast %[[ADDRESS_OF_6]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[DECLARE_6:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_2]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEvolatile_array_pointer"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[ADDRESS_OF_7:.*]] = fir.address_of(@_QFEvolatile_array_target) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VOLATILE_CAST_3:.*]] = fir.volatile_cast %[[ADDRESS_OF_7]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[DECLARE_7:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_3]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEvolatile_array_target"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[ALLOCA_2:.*]] = fir.alloca i32 {bindc_name = "volatile_integer", uniq_name = "_QFEvolatile_integer"}
! CHECK:           %[[VOLATILE_CAST_4:.*]] = fir.volatile_cast %[[ALLOCA_2]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[DECLARE_8:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_4]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEvolatile_integer"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[ALLOCA_3:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "volatile_integer_pointer", uniq_name = "_QFEvolatile_integer_pointer"}
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           fir.store %[[EMBOX_0]] to %[[ALLOCA_3]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VOLATILE_CAST_5:.*]] = fir.volatile_cast %[[ALLOCA_3]] : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>
! CHECK:           %[[DECLARE_9:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_5]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEvolatile_integer_pointer"} : (!fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>)
! CHECK:           %[[ADDRESS_OF_8:.*]] = fir.address_of(@_QFEvolatile_integer_target) : !fir.ref<i32>
! CHECK:           %[[VOLATILE_CAST_6:.*]] = fir.volatile_cast %[[ADDRESS_OF_8]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[DECLARE_10:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_6]] {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEvolatile_integer_target"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_7]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<?xi32>, volatile>
! CHECK:           %[[EMBOX_1:.*]] = fir.embox %[[CONVERT_0]](%[[SHAPE_0]]) : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[EMBOX_1]] to %[[DECLARE_6]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VOLATILE_CAST_7:.*]] = fir.volatile_cast %[[DECLARE_2]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[VOLATILE_CAST_7]] : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<?xi32>, volatile>
! CHECK:           %[[EMBOX_2:.*]] = fir.embox %[[CONVERT_1]](%[[SHAPE_0]]) : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[EMBOX_2]] to %[[DECLARE_6]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[ZERO_BITS_1:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[SHAPE_2:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[EMBOX_3:.*]] = fir.embox %[[ZERO_BITS_1]](%[[SHAPE_2]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[EMBOX_3]] to %[[ALLOCA_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>
! CHECK:           %[[DECLARE_11:.*]]:2 = hlfir.declare %[[ALLOCA_1]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>)
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_11]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>>
! CHECK:           fir.store %[[LOAD_0]] to %[[DECLARE_6]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VOLATILE_CAST_8:.*]] = fir.volatile_cast %[[DECLARE_7]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[VOLATILE_CAST_8]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[EMBOX_4:.*]] = fir.embox %[[CONVERT_2]](%[[SHAPE_0]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.store %[[EMBOX_4]] to %[[DECLARE_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[DECLARE_2]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[EMBOX_5:.*]] = fir.embox %[[CONVERT_3]](%[[SHAPE_0]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.store %[[EMBOX_5]] to %[[DECLARE_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[EMBOX_6:.*]] = fir.embox %[[DECLARE_10]]#0 : (!fir.ref<i32, volatile>) -> !fir.box<!fir.ptr<i32>, volatile>
! CHECK:           fir.store %[[EMBOX_6]] to %[[DECLARE_9]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>
! CHECK:           %[[EMBOX_7:.*]] = fir.embox %[[ZERO_BITS_0]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>, volatile>
! CHECK:           fir.store %[[EMBOX_7]] to %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.ptr<i32>, volatile>>
! CHECK:           %[[DECLARE_12:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.box<!fir.ptr<i32>, volatile>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>, volatile>>, !fir.ref<!fir.box<!fir.ptr<i32>, volatile>>)
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_12]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>, volatile>>
! CHECK:           fir.store %[[LOAD_1]] to %[[DECLARE_9]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>, volatile>, volatile>
! CHECK:           %[[VOLATILE_CAST_9:.*]] = fir.volatile_cast %[[DECLARE_4]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           fir.call @_QFPsub_nonvolatile_array(%[[VOLATILE_CAST_9]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[EMBOX_8:.*]] = fir.embox %[[DECLARE_4]]#0(%[[SHAPE_0]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VOLATILE_CAST_10:.*]] = fir.volatile_cast %[[EMBOX_8]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[VOLATILE_CAST_10]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[CONVERT_4]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           fir.call @_QFPsub_volatile_array(%[[VOLATILE_CAST_9]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[EMBOX_9:.*]] = fir.embox %[[DECLARE_0]]#0(%[[SHAPE_0]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[EMBOX_9]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[CONVERT_5]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           fir.call @_QFPsub_volatile_array(%[[DECLARE_0]]#0) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[DECLARE_6]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
! CHECK:           fir.call @_QFPsub_volatile_array_pointer(%[[CONVERT_6]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
! CHECK:           %[[VOLATILE_CAST_11:.*]] = fir.volatile_cast %[[DECLARE_1]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
! CHECK:           fir.call @_QFPsub_volatile_array_pointer(%[[VOLATILE_CAST_11]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_4]]#0 (%[[CONSTANT_0]]:%[[CONSTANT_2]]:%[[CONSTANT_0]])  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<10xi32>, volatile>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[EMBOX_10:.*]] = fir.embox %[[DESIGNATE_0]](%[[SHAPE_0]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VOLATILE_CAST_12:.*]] = fir.volatile_cast %[[EMBOX_10]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[VOLATILE_CAST_12]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape(%[[CONVERT_7]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[DECLARE_5]]#0 (%[[CONSTANT_0]]:%[[CONSTANT_2]]:%[[CONSTANT_0]], %[[CONSTANT_0]]:%[[CONSTANT_2]]:%[[CONSTANT_0]])  shape %[[SHAPE_1]] : (!fir.ref<!fir.array<10x10xi32>, volatile>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>, volatile>
! CHECK:           %[[VOLATILE_CAST_13:.*]] = fir.volatile_cast %[[DESIGNATE_1]] : (!fir.box<!fir.array<10x10xi32>, volatile>) -> !fir.box<!fir.array<10x10xi32>>
! CHECK:           %[[CONVERT_8:.*]] = fir.convert %[[VOLATILE_CAST_13]] : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK:           fir.call @_QFPsub_volatile_array_assumed_shape_2d(%[[CONVERT_8]]) fastmath<contract> : (!fir.box<!fir.array<?x?xi32>>) -> ()
! CHECK:           %[[CONVERT_9:.*]] = fir.convert %[[VOLATILE_CAST_10]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QFPsub_select_rank(%[[CONVERT_9]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()
! CHECK:           %[[EMBOX_11:.*]] = fir.embox %[[DECLARE_5]]#0(%[[SHAPE_1]]) : (!fir.ref<!fir.array<10x10xi32>, volatile>, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>, volatile>
! CHECK:           %[[VOLATILE_CAST_14:.*]] = fir.volatile_cast %[[EMBOX_11]] : (!fir.box<!fir.array<10x10xi32>, volatile>) -> !fir.box<!fir.array<10x10xi32>>
! CHECK:           %[[CONVERT_10:.*]] = fir.convert %[[VOLATILE_CAST_14]] : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QFPsub_select_rank(%[[CONVERT_10]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_nonvolatile_array(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : i32
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFFsub_nonvolatile_arrayEarr"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[CONSTANT_0]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DESIGNATE_0]] : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_volatile_array_assumed_shape(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr"}) {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : i32
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ARG0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>, volatile>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_volatile_array_assumed_shapeEarr"} : (!fir.box<!fir.array<?xi32>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>, volatile>, !fir.box<!fir.array<?xi32>, volatile>)
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[CONSTANT_0]])  : (!fir.box<!fir.array<?xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DESIGNATE_0]] : i32, !fir.ref<i32, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_volatile_array_assumed_shape_2d(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "arr"}) {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : i32
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ARG0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>, volatile>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_volatile_array_assumed_shape_2dEarr"} : (!fir.box<!fir.array<?x?xi32>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>, volatile>, !fir.box<!fir.array<?x?xi32>, volatile>)
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[CONSTANT_0]], %[[CONSTANT_0]])  : (!fir.box<!fir.array<?x?xi32>, volatile>, index, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DESIGNATE_0]] : i32, !fir.ref<i32, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_volatile_array(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : i32
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ARG0]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_volatile_arrayEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[CONSTANT_0]])  : (!fir.ref<!fir.array<10xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DESIGNATE_0]] : i32, !fir.ref<i32, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_volatile_array_pointer(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile> {fir.bindc_name = "arr"}) {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : i32
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ARG0]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFFsub_volatile_array_pointerEarr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_0]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[LOAD_0]] (%[[CONSTANT_0]])  : (!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DESIGNATE_0]] : i32, !fir.ref<i32, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPsub_select_rank(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<*:i32>> {fir.bindc_name = "arr"}) {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : i32
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : i8
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i8
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ARG0]] : (!fir.box<!fir.array<*:i32>>) -> !fir.box<!fir.array<*:i32>, volatile>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_select_rankEarr"} : (!fir.box<!fir.array<*:i32>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<*:i32>, volatile>, !fir.box<!fir.array<*:i32>, volatile>)
! CHECK:           %[[VOLATILE_CAST_1:.*]] = fir.volatile_cast %[[DECLARE_0]]#0 : (!fir.box<!fir.array<*:i32>, volatile>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[VOLATILE_CAST_1]] : (!fir.box<!fir.array<*:i32>>) -> !fir.box<none>
! CHECK:           %[[CALL_0:.*]] = fir.call @_FortranAIsAssumedSize(%[[CONVERT_0]]) : (!fir.box<none>) -> i1
! CHECK:           cf.cond_br %[[CALL_0]], ^bb4, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[BOX_RANK_0:.*]] = fir.box_rank %[[DECLARE_0]]#0 : (!fir.box<!fir.array<*:i32>, volatile>) -> i8
! CHECK:           fir.select_case %[[BOX_RANK_0]] : i8 [#fir.point, %[[CONSTANT_3]], ^bb2, #fir.point, %[[CONSTANT_2]], ^bb3, unit, ^bb4]
! CHECK:         ^bb2:
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[DECLARE_0]]#0 : (!fir.box<!fir.array<*:i32>, volatile>) -> !fir.box<!fir.array<?xi32>, volatile>
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[CONVERT_1]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_select_rankEarr"} : (!fir.box<!fir.array<?xi32>, volatile>) -> (!fir.box<!fir.array<?xi32>, volatile>, !fir.box<!fir.array<?xi32>, volatile>)
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_1]]#0 (%[[CONSTANT_0]])  : (!fir.box<!fir.array<?xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DESIGNATE_0]] : i32, !fir.ref<i32, volatile>
! CHECK:           cf.br ^bb4
! CHECK:         ^bb3:
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[DECLARE_0]]#0 : (!fir.box<!fir.array<*:i32>, volatile>) -> !fir.box<!fir.array<?x?x?x?xi32>, volatile>
! CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[CONVERT_2]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFsub_select_rankEarr"} : (!fir.box<!fir.array<?x?x?x?xi32>, volatile>) -> (!fir.box<!fir.array<?x?x?x?xi32>, volatile>, !fir.box<!fir.array<?x?x?x?xi32>, volatile>)
! CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[DECLARE_2]]#0 (%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]])  : (!fir.box<!fir.array<?x?x?x?xi32>, volatile>, index, index, index, index) -> !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DESIGNATE_1]] : i32, !fir.ref<i32, volatile>
! CHECK:           cf.br ^bb4
! CHECK:         ^bb4:
! CHECK:           return
! CHECK:         }
