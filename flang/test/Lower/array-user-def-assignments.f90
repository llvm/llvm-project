! Test lower of elemental user defined assignments
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

module defined_assignments
  type t
    integer :: i
  end type
  interface assignment(=)
    elemental subroutine assign_t(a,b)
      import t
      type(t),intent(out) :: a
      type(t),intent(in) :: b
    end
  end interface
  interface assignment(=)
    elemental subroutine assign_logical_to_real(a,b)
      real, intent(out) :: a
      logical, intent(in) :: b
    end
  end interface
  interface assignment(=)
    elemental subroutine assign_real_to_logical(a,b)
      logical, intent(out) :: a
      real, intent(in) :: b
    end
  end interface
end module

! CHECK-LABEL: func @_QPtest_derived(
! CHECK-SAME:        %arg0: !fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>> {fir.bindc_name = "x"}) {
! CHECK:     %[[C_100:[-0-9a-z_]+]] = arith.constant 100 : index
! CHECK:     %[[V_0:[0-9]+]] = fir.shape %[[C_100]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_1:[0-9]+]] = fir.array_load %arg0(%[[V_0]]) : (!fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>, !fir.shape<1>) -> !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:     %[[C_100_i64:[-0-9a-z_]+]] = arith.constant 100 : i64
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_100_i64]] : (i64) -> index
! CHECK:     %[[C_m1_i64:[-0-9a-z_]+]] = arith.constant -1 : i64
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_m1_i64]] : (i64) -> index
! CHECK:     %[[C_1_i64:[-0-9a-z_]+]] = arith.constant 1 : i64
! CHECK:     %[[V_4:[0-9]+]] = fir.convert %[[C_1_i64]] : (i64) -> index
! CHECK:     %[[V_5:[0-9]+]] = fir.shape %[[C_100]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_6:[0-9]+]] = fir.slice %[[V_2]], %[[V_4]], %[[V_3:[0-9]+]] : (index, index, index) -> !fir.slice<1>
! CHECK:     %[[V_7:[0-9]+]] = fir.array_load %arg0(%[[V_5]]) [%[[V_6]]] : (!fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_8:[0-9]+]] = arith.subi %[[C_100]], %[[C_1]] : index
! CHECK:     %[[V_9:[0-9]+]] = fir.do_loop %arg1 = %[[C_0]] to %[[V_8]] step %[[C_1]] unordered iter_args(%arg2 = %[[V_1]]) -> (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>) {
! CHECK:       %[[V_10:[0-9]+]] = fir.array_access %[[V_7]], %arg1 : (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, index) -> !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:       %[[V_11:[0-9]+]] = fir.no_reassoc %[[V_10:[0-9]+]] : !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:       %[[V_12:[0-9]+]]:2 = fir.array_modify %arg2, %arg1 : (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, index) -> (!fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>)
! CHECK:       fir.call @_QPassign_t(%[[V_12]]#0, %[[V_11]]) fastmath<contract> : (!fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>) -> ()
! CHECK:       fir.result %[[V_12]]#1 : !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_1]], %[[V_9]] to %arg0 : !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPtest_intrinsic(
! CHECK-SAME:                          %arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "x"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca !fir.logical<4>
! CHECK:     %[[C_100:[-0-9a-z_]+]] = arith.constant 100 : index
! CHECK:     %[[V_1:[0-9]+]] = fir.shape %[[C_100]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_2:[0-9]+]] = fir.array_load %arg0(%[[V_1]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:     %[[C_100_i64:[-0-9a-z_]+]] = arith.constant 100 : i64
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_100_i64]] : (i64) -> index
! CHECK:     %[[C_m1_i64:[-0-9a-z_]+]] = arith.constant -1 : i64
! CHECK:     %[[V_4:[0-9]+]] = fir.convert %[[C_m1_i64]] : (i64) -> index
! CHECK:     %[[C_1_i64:[-0-9a-z_]+]] = arith.constant 1 : i64
! CHECK:     %[[V_5:[0-9]+]] = fir.convert %[[C_1_i64]] : (i64) -> index
! CHECK:     %[[V_6:[0-9]+]] = fir.shape %[[C_100]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_7:[0-9]+]] = fir.slice %[[V_3]], %[[V_5]], %[[V_4:[0-9]+]] : (index, index, index) -> !fir.slice<1>
! CHECK:     %[[V_8:[0-9]+]] = fir.array_load %arg0(%[[V_6]]) [%[[V_7]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
! CHECK:     %[[C_st:[-0-9a-z_]+]] = arith.constant 0.000000e+00 : f32
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_9:[0-9]+]] = arith.subi %[[C_100]], %[[C_1]] : index
! CHECK:     %[[V_10:[0-9]+]] = fir.do_loop %arg1 = %[[C_0]] to %[[V_9]] step %[[C_1]] unordered iter_args(%arg2 = %[[V_2]]) -> (!fir.array<100xf32>) {
! CHECK:       %[[V_11:[0-9]+]] = fir.array_fetch %[[V_8]], %arg1 : (!fir.array<100xf32>, index) -> f32
! CHECK:       %[[V_12:[0-9]+]] = arith.cmpf olt, %[[V_11]], %[[C_st]] {{.*}} : f32
! CHECK:       %[[V_13:[0-9]+]]:2 = fir.array_modify %arg2, %arg1 : (!fir.array<100xf32>, index) -> (!fir.ref<f32>, !fir.array<100xf32>)
! CHECK:       %[[V_14:[0-9]+]] = fir.convert %[[V_12:[0-9]+]] : (i1) -> !fir.logical<4>
! CHECK:       fir.store %[[V_14]] to %[[V_0:[0-9]+]] : !fir.ref<!fir.logical<4>>
! CHECK:       fir.call @_QPassign_logical_to_real(%[[V_13]]#0, %[[V_0]]) fastmath<contract> : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:       fir.result %[[V_13]]#1 : !fir.array<100xf32>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_2]], %[[V_10]] to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPtest_intrinsic_2(
! CHECK-SAME:                            %arg0: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca f32
! CHECK:     %[[C_100:[-0-9a-z_]+]] = arith.constant 100 : index
! CHECK:     %[[C_100_0:[-0-9a-z_]+]] = arith.constant 100 : index
! CHECK:     %[[V_1:[0-9]+]] = fir.shape %[[C_100]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_2:[0-9]+]] = fir.array_load %arg0(%[[V_1]]) : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<100x!fir.logical<4>>
! CHECK:     %[[V_3:[0-9]+]] = fir.shape %[[C_100_0]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_4:[0-9]+]] = fir.array_load %arg1(%[[V_3]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_5:[0-9]+]] = arith.subi %[[C_100]], %[[C_1]] : index
! CHECK:     %[[V_6:[0-9]+]] = fir.do_loop %arg2 = %[[C_0]] to %[[V_5]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_2]]) -> (!fir.array<100x!fir.logical<4>>) {
! CHECK:       %[[V_7:[0-9]+]] = fir.array_fetch %[[V_4]], %arg2 : (!fir.array<100xf32>, index) -> f32
! CHECK:       %[[V_8:[0-9]+]] = fir.no_reassoc %[[V_7:[0-9]+]] : f32
! CHECK:       %[[V_9:[0-9]+]]:2 = fir.array_modify %arg3, %arg2 : (!fir.array<100x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<100x!fir.logical<4>>)
! CHECK:       fir.store %[[V_8]] to %[[V_0:[0-9]+]] : !fir.ref<f32>
! CHECK:       fir.call @_QPassign_real_to_logical(%[[V_9]]#0, %[[V_0]]) fastmath<contract> : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:       fir.result %[[V_9]]#1 : !fir.array<100x!fir.logical<4>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_2]], %[[V_6]] to %arg0 : !fir.array<100x!fir.logical<4>>, !fir.array<100x!fir.logical<4>>, !fir.ref<!fir.array<100x!fir.logical<4>>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPfrom_char(
! CHECK-SAME:                     %arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"}, %arg1: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_0:[0-9]+]]:3 = fir.box_dims %arg0, %[[C_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:     %[[V_1:[0-9]+]] = fir.array_load %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:     %[[V_2:[0-9]+]] = fir.array_load %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:     %[[V_3:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_4:[0-9]+]] = arith.divsi %[[V_3]], %[[C_1]] : index
! CHECK:     %[[C_1_0:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0_1:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_5:[0-9]+]] = arith.subi %[[V_0]]#1, %[[C_1_0]] : index
! CHECK:     %[[V_6:[0-9]+]] = fir.do_loop %arg2 = %[[C_0_1]] to %[[V_5]] step %[[C_1_0]] unordered iter_args(%arg3 = %[[V_1]]) -> (!fir.array<?xi32>) {
! CHECK:       %[[V_7:[0-9]+]] = fir.array_access %[[V_2]], %arg2 typeparams %[[V_4:[0-9]+]] : (!fir.array<?x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:       %[[V_8:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:       %[[V_9:[0-9]+]] = fir.no_reassoc %[[V_7:[0-9]+]] : !fir.ref<!fir.char<1,?>>
! CHECK:       %[[V_10:[0-9]+]]:2 = fir.array_modify %arg3, %arg2 : (!fir.array<?xi32>, index) -> (!fir.ref<i32>, !fir.array<?xi32>)
! CHECK:       %[[V_11:[0-9]+]] = fir.emboxchar %[[V_9]], %[[V_8:[0-9]+]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:       fir.call @_QPsfrom_char(%[[V_10]]#0, %[[V_11]]) fastmath<contract> : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:       fir.result %[[V_10]]#1 : !fir.array<?xi32>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_1]], %[[V_6]] to %arg0 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPto_char(
! CHECK-SAME:                   %arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"}, %arg1: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca i32
! CHECK:     %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_1:[0-9]+]]:3 = fir.box_dims %arg1, %[[C_0]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:     %[[V_2:[0-9]+]] = fir.array_load %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:     %[[V_3:[0-9]+]] = fir.array_load %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_4:[0-9]+]] = arith.subi %[[V_1]]#1, %[[C_1]] : index
! CHECK:     %[[V_5:[0-9]+]] = fir.do_loop %arg2 = %[[C_0_0]] to %[[V_4]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_2]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:       %[[V_6:[0-9]+]] = fir.array_fetch %[[V_3]], %arg2 : (!fir.array<?xi32>, index) -> i32
! CHECK:       %[[V_7:[0-9]+]] = fir.no_reassoc %[[V_6:[0-9]+]] : i32
! CHECK:       %[[V_8:[0-9]+]]:2 = fir.array_modify %arg3, %arg2 : (!fir.array<?x!fir.char<1,?>>, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>)
! CHECK:       %[[V_9:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:       %[[V_10:[0-9]+]] = fir.emboxchar %[[V_8]]#0, %[[V_9:[0-9]+]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:       fir.store %[[V_7]] to %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       fir.call @_QPsto_char(%[[V_10]], %[[V_0]]) fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:       fir.result %[[V_8]]#1 : !fir.array<?x!fir.char<1,?>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_2]], %[[V_5]] to %arg1 : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:     return
! CHECK:   }

subroutine test_derived(x)
  use defined_assignments
  type(t) :: x(100)
  x = x(100:1:-1)
end subroutine

subroutine test_intrinsic(x)
  use defined_assignments
  real :: x(100)
  x = x(100:1:-1) .lt. 0.
end subroutine

subroutine test_intrinsic_2(x, y)
  use defined_assignments
  logical :: x(100)
  real :: y(100)
  x = y
end subroutine

subroutine from_char(i, c)
  interface assignment(=)
    elemental subroutine sfrom_char(a,b)
      integer, intent(out) :: a
      character(*),intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  i = c
end subroutine

subroutine to_char(i, c)
  interface assignment(=)
    elemental subroutine sto_char(a,b)
      character(*), intent(out) :: a
      integer,intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  c = i
end subroutine

! -----------------------------------------------------------------------------
!     Test user defined assignments inside FORALL and WHERE
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_in_forall_1(
! CHECK-SAME:                            %arg0: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca f32
! CHECK:     %[[V_1:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:     %[[C_10:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_10_0:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_4:[0-9]+]] = fir.shape %[[C_10]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %arg0(%[[V_4]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:     %[[V_6:[0-9]+]] = fir.shape %[[C_10_0]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_7:[0-9]+]] = fir.array_load %arg1(%[[V_6]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:     %[[V_8:[0-9]+]] = fir.do_loop %arg2 = %[[V_2]] to %[[V_3]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_5]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:       %[[V_9:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_9]] to %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[C_1_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_10:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_11:[0-9]+]] = fir.convert %[[V_10:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_12:[0-9]+]] = fir.convert %[[V_11:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_13:[0-9]+]] = arith.subi %[[V_12]], %[[C_1_1]] : index
! CHECK:       %[[V_14:[0-9]+]] = fir.array_fetch %[[V_7]], %[[V_13:[0-9]+]] : (!fir.array<10xf32>, index) -> f32
! CHECK:       %[[V_15:[0-9]+]] = fir.no_reassoc %[[V_14:[0-9]+]] : f32
! CHECK:       %[[C_1_2:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_16:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_17:[0-9]+]] = fir.convert %[[V_16:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_18:[0-9]+]] = fir.convert %[[V_17:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_19:[0-9]+]] = arith.subi %[[V_18]], %[[C_1_2]] : index
! CHECK:       %[[V_20:[0-9]+]]:2 = fir.array_modify %arg3, %[[V_19:[0-9]+]] : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:       fir.store %[[V_15]] to %[[V_0:[0-9]+]] : !fir.ref<f32>
! CHECK:       fir.call @_QPassign_real_to_logical(%[[V_20]]#0, %[[V_0]]) fastmath<contract> : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:       fir.result %[[V_20]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_5]], %[[V_8]] to %arg0 : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPtest_in_forall_2(
! CHECK-SAME:                            %arg0: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca !fir.logical<4>
! CHECK:     %[[V_1:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:     %[[C_10:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_4:[0-9]+]] = fir.shape %[[C_10]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %arg1(%[[V_4]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:     %[[V_6:[0-9]+]] = fir.do_loop %arg2 = %[[V_2]] to %[[V_3]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_5]]) -> (!fir.array<10xf32>) {
! CHECK:       %[[V_7:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_7]] to %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[C_1_0:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_8:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_9:[0-9]+]] = fir.convert %[[V_8:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_10:[0-9]+]] = fir.convert %[[V_9:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_11:[0-9]+]] = arith.subi %[[V_10]], %[[C_1_0]] : index
! CHECK:       %[[V_12:[0-9]+]] = fir.array_fetch %[[V_5]], %[[V_11:[0-9]+]] : (!fir.array<10xf32>, index) -> f32
! CHECK:       %[[C_st:[-0-9a-z_]+]] = arith.constant 0.000000e+00 : f32
! CHECK:       %[[V_13:[0-9]+]] = arith.cmpf olt, %[[V_12]], %[[C_st]] {{.*}} : f32
! CHECK:       %[[C_1_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_14:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_15:[0-9]+]] = fir.convert %[[V_14:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_16:[0-9]+]] = fir.convert %[[V_15:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_17:[0-9]+]] = arith.subi %[[V_16]], %[[C_1_1]] : index
! CHECK:       %[[V_18:[0-9]+]]:2 = fir.array_modify %arg3, %[[V_17:[0-9]+]] : (!fir.array<10xf32>, index) -> (!fir.ref<f32>, !fir.array<10xf32>)
! CHECK:       %[[V_19:[0-9]+]] = fir.convert %[[V_13:[0-9]+]] : (i1) -> !fir.logical<4>
! CHECK:       fir.store %[[V_19]] to %[[V_0:[0-9]+]] : !fir.ref<!fir.logical<4>>
! CHECK:       fir.call @_QPassign_logical_to_real(%[[V_18]]#0, %[[V_0]]) fastmath<contract> : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:       fir.result %[[V_18]]#1 : !fir.array<10xf32>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_5]], %[[V_6]] to %arg1 : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPtest_intrinsic_where_1(
! CHECK-SAME:             %arg0: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "l"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca f32
! CHECK:     %[[C_10:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_10_0:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_10_1:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_10_2:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[V_1:[0-9]+]] = fir.shape %[[C_10]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_2:[0-9]+]] = fir.array_load %arg2(%[[V_1]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:     %[[V_3:[0-9]+]] = fir.allocmem !fir.array<10x!fir.logical<4>>
! CHECK:     %[[V_4:[0-9]+]] = fir.shape %[[C_10_2]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %[[V_3]](%[[V_4]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_6:[0-9]+]] = arith.subi %[[C_10_2]], %[[C_1]] : index
! CHECK:     %[[V_7:[0-9]+]] = fir.do_loop %arg3 = %[[C_0]] to %[[V_6]] step %[[C_1]] unordered iter_args(%arg4 = %[[V_5]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:       %[[V_15:[0-9]+]] = fir.array_fetch %[[V_2]], %arg3 : (!fir.array<10x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:       %[[V_16:[0-9]+]] = fir.array_update %arg4, %[[V_15]], %arg3 : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
! CHECK:       fir.result %[[V_16:[0-9]+]] : !fir.array<10x!fir.logical<4>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_5]], %[[V_7]] to %[[V_3:[0-9]+]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:     %[[V_8:[0-9]+]] = fir.shape %[[C_10_2]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_9:[0-9]+]] = fir.shape %[[C_10_0]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_10:[0-9]+]] = fir.array_load %arg0(%[[V_9]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:     %[[V_11:[0-9]+]] = fir.shape %[[C_10_1]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_12:[0-9]+]] = fir.array_load %arg1(%[[V_11]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:     %[[C_1_3:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0_4:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_13:[0-9]+]] = arith.subi %[[C_10_0]], %[[C_1_3]] : index
! CHECK:     %[[V_14:[0-9]+]] = fir.do_loop %arg3 = %[[C_0_4]] to %[[V_13]] step %[[C_1_3]] unordered iter_args(%arg4 = %[[V_10]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:       %[[C_1_5:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_15:[0-9]+]] = arith.addi %arg3, %[[C_1_5]] : index
! CHECK:       %[[V_16:[0-9]+]] = fir.array_coor %[[V_3]](%[[V_8]]) %[[V_15:[0-9]+]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:       %[[V_17:[0-9]+]] = fir.load %[[V_16:[0-9]+]] : !fir.ref<!fir.logical<4>>
! CHECK:       %[[V_18:[0-9]+]] = fir.convert %[[V_17:[0-9]+]] : (!fir.logical<4>) -> i1
! CHECK:       %[[V_19:[0-9]+]] = fir.if %[[V_18]] -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:         %[[V_20:[0-9]+]] = fir.array_fetch %[[V_12]], %arg3 : (!fir.array<10xf32>, index) -> f32
! CHECK:         %[[V_21:[0-9]+]] = fir.no_reassoc %[[V_20:[0-9]+]] : f32
! CHECK:         %[[V_22:[0-9]+]]:2 = fir.array_modify %arg4, %arg3 : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:         fir.store %[[V_21]] to %[[V_0:[0-9]+]] : !fir.ref<f32>
! CHECK:         fir.call @_QPassign_real_to_logical(%[[V_22]]#0, %[[V_0]]) fastmath<contract> : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:         fir.result %[[V_22]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:       } else {
! CHECK:         fir.result %arg4 : !fir.array<10x!fir.logical<4>>
! CHECK:       }
! CHECK:       fir.result %[[V_19:[0-9]+]] : !fir.array<10x!fir.logical<4>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_10]], %[[V_14]] to %arg0 : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:     fir.freemem %[[V_3:[0-9]+]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPtest_intrinsic_where_2(
! CHECK-SAME:           %arg0: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "l"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca !fir.logical<4>
! CHECK:     %[[C_10:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_10_0:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_10_1:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[V_1:[0-9]+]] = fir.shape %[[C_10]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_2:[0-9]+]] = fir.array_load %arg2(%[[V_1]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:     %[[V_3:[0-9]+]] = fir.allocmem !fir.array<10x!fir.logical<4>>
! CHECK:     %[[V_4:[0-9]+]] = fir.shape %[[C_10_1]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %[[V_3]](%[[V_4]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_6:[0-9]+]] = arith.subi %[[C_10_1]], %[[C_1]] : index
! CHECK:     %[[V_7:[0-9]+]] = fir.do_loop %arg3 = %[[C_0]] to %[[V_6]] step %[[C_1]] unordered iter_args(%arg4 = %[[V_5]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:       %[[V_15:[0-9]+]] = fir.array_fetch %[[V_2]], %arg3 : (!fir.array<10x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:       %[[V_16:[0-9]+]] = fir.array_update %arg4, %[[V_15]], %arg3 : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
! CHECK:       fir.result %[[V_16:[0-9]+]] : !fir.array<10x!fir.logical<4>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_5]], %[[V_7]] to %[[V_3:[0-9]+]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:     %[[V_8:[0-9]+]] = fir.shape %[[C_10_1]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_9:[0-9]+]] = fir.shape %[[C_10_0]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_10:[0-9]+]] = fir.array_load %arg1(%[[V_9]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:     %[[V_11:[0-9]+]] = fir.shape %[[C_10_0]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_12:[0-9]+]] = fir.array_load %arg1(%[[V_11]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:     %[[C_st:[-0-9a-z_]+]] = arith.constant 0.000000e+00 : f32
! CHECK:     %[[C_1_2:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[C_0_3:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:     %[[V_13:[0-9]+]] = arith.subi %[[C_10_0]], %[[C_1_2]] : index
! CHECK:     %[[V_14:[0-9]+]] = fir.do_loop %arg3 = %[[C_0_3]] to %[[V_13]] step %[[C_1_2]] unordered iter_args(%arg4 = %[[V_10]]) -> (!fir.array<10xf32>) {
! CHECK:       %[[C_1_4:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_15:[0-9]+]] = arith.addi %arg3, %[[C_1_4]] : index
! CHECK:       %[[V_16:[0-9]+]] = fir.array_coor %[[V_3]](%[[V_8]]) %[[V_15:[0-9]+]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:       %[[V_17:[0-9]+]] = fir.load %[[V_16:[0-9]+]] : !fir.ref<!fir.logical<4>>
! CHECK:       %[[V_18:[0-9]+]] = fir.convert %[[V_17:[0-9]+]] : (!fir.logical<4>) -> i1
! CHECK:       %[[V_19:[0-9]+]] = fir.if %[[V_18]] -> (!fir.array<10xf32>) {
! CHECK:         %[[V_20:[0-9]+]] = fir.array_fetch %[[V_12]], %arg3 : (!fir.array<10xf32>, index) -> f32
! CHECK:         %[[V_21:[0-9]+]] = arith.cmpf olt, %[[V_20]], %[[C_st]] {{.*}} : f32
! CHECK:         %[[V_22:[0-9]+]]:2 = fir.array_modify %arg4, %arg3 : (!fir.array<10xf32>, index) -> (!fir.ref<f32>, !fir.array<10xf32>)
! CHECK:         %[[V_23:[0-9]+]] = fir.convert %[[V_21:[0-9]+]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[V_23]] to %[[V_0:[0-9]+]] : !fir.ref<!fir.logical<4>>
! CHECK:         fir.call @_QPassign_logical_to_real(%[[V_22]]#0, %[[V_0]]) fastmath<contract> : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:         fir.result %[[V_22]]#1 : !fir.array<10xf32>
! CHECK:       } else {
! CHECK:         fir.result %arg4 : !fir.array<10xf32>
! CHECK:       }
! CHECK:       fir.result %[[V_19:[0-9]+]] : !fir.array<10xf32>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_10]], %[[V_14]] to %arg1 : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:     fir.freemem %[[V_3:[0-9]+]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPtest_scalar_func_but_not_elemental(
! CHECK-SAME:        %arg0: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca i32
! CHECK:     %[[V_1:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:     %[[C_100:[-0-9a-z_]+]] = arith.constant 100 : index
! CHECK:     %[[C_100_0:[-0-9a-z_]+]] = arith.constant 100 : index
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_4:[0-9]+]] = fir.shape %[[C_100]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %arg0(%[[V_4]]) : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<100x!fir.logical<4>>
! CHECK:     %[[V_6:[0-9]+]] = fir.shape %[[C_100_0]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_7:[0-9]+]] = fir.array_load %arg1(%[[V_6]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
! CHECK:     %[[V_8:[0-9]+]] = fir.do_loop %arg2 = %[[V_2]] to %[[V_3]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_5]]) -> (!fir.array<100x!fir.logical<4>>) {
! CHECK:       %[[V_9:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_9]] to %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[C_1_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_10:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_11:[0-9]+]] = fir.convert %[[V_10:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_12:[0-9]+]] = fir.convert %[[V_11:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_13:[0-9]+]] = arith.subi %[[V_12]], %[[C_1_1]] : index
! CHECK:       %[[V_14:[0-9]+]] = fir.array_fetch %[[V_7]], %[[V_13:[0-9]+]] : (!fir.array<100xi32>, index) -> i32
! CHECK:       %[[V_15:[0-9]+]] = fir.no_reassoc %[[V_14:[0-9]+]] : i32
! CHECK:       %[[C_1_2:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_16:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_17:[0-9]+]] = fir.convert %[[V_16:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_18:[0-9]+]] = fir.convert %[[V_17:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_19:[0-9]+]] = arith.subi %[[V_18]], %[[C_1_2]] : index
! CHECK:       %[[V_20:[0-9]+]]:2 = fir.array_modify %arg3, %[[V_19:[0-9]+]] : (!fir.array<100x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<100x!fir.logical<4>>)
! CHECK:       fir.store %[[V_15]] to %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       fir.call @_QPassign_integer_to_logical(%[[V_20]]#0, %[[V_0]]) fastmath<contract> : (!fir.ref<!fir.logical<4>>, !fir.ref<i32>) -> ()
! CHECK:       fir.result %[[V_20]]#1 : !fir.array<100x!fir.logical<4>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_5]], %[[V_8]] to %arg0 : !fir.array<100x!fir.logical<4>>, !fir.array<100x!fir.logical<4>>, !fir.ref<!fir.array<100x!fir.logical<4>>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPtest_in_forall_with_cleanup(
! CHECK-SAME:       %arg0: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = ".result"}
! CHECK:     %[[V_1:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:     %[[C_10:[-0-9a-z_]+]] = arith.constant 10 : index
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_4:[0-9]+]] = fir.shape %[[C_10]] : (index) -> !fir.shape<1>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %arg0(%[[V_4]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:     %[[V_6:[0-9]+]] = fir.do_loop %arg2 = %[[V_2]] to %[[V_3]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_5]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:       %[[V_7:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_7]] to %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_8:[0-9]+]] = fir.call @_QPreturns_alloc(%[[V_1]]) fastmath<contract> : (!fir.ref<i32>) -> !fir.box<!fir.heap<f32>>
! CHECK:       fir.save_result %[[V_8]] to %[[V_0:[0-9]+]] : !fir.box<!fir.heap<f32>>, !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:       %[[V_9:[0-9]+]] = fir.load %[[V_0:[0-9]+]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:       %[[V_10:[0-9]+]] = fir.box_addr %[[V_9:[0-9]+]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:       %[[C_1_0:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_11:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_12:[0-9]+]] = fir.convert %[[V_11:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_13:[0-9]+]] = fir.convert %[[V_12:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_14:[0-9]+]] = arith.subi %[[V_13]], %[[C_1_0]] : index
! CHECK:       %[[V_15:[0-9]+]]:2 = fir.array_modify %arg3, %[[V_14:[0-9]+]] : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:       %[[V_16:[0-9]+]] = fir.convert %[[V_10:[0-9]+]] : (!fir.heap<f32>) -> !fir.ref<f32>
! CHECK:       fir.call @_QPassign_real_to_logical(%[[V_15]]#0, %[[V_16]]) fastmath<contract> : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:       %[[V_17:[0-9]+]] = fir.load %[[V_0:[0-9]+]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:       %[[V_18:[0-9]+]] = fir.box_addr %[[V_17:[0-9]+]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:       %[[V_19:[0-9]+]] = fir.convert %[[V_18:[0-9]+]] : (!fir.heap<f32>) -> i64
! CHECK:       %[[C_0_i64:[-0-9a-z_]+]] = arith.constant 0 : i64
! CHECK:       %[[V_20:[0-9]+]] = arith.cmpi ne, %[[V_19]], %[[C_0_i64]] : i64
! CHECK:       fir.if %[[V_20]] {
! CHECK:         fir.freemem %[[V_18:[0-9]+]] : !fir.heap<f32>
! CHECK:       }
! CHECK:       fir.result %[[V_15]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_5]], %[[V_6]] to %arg0 : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:     return
! CHECK:   }

subroutine test_in_forall_1(x, y)
  use defined_assignments
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) x(i) = y(i)
end subroutine

subroutine test_in_forall_2(x, y)
  use defined_assignments
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) y(i) = y(i).lt.0.
end subroutine

subroutine test_intrinsic_where_1(x, y, l)
  use defined_assignments
  logical :: x(10), l(10)
  real :: y(10)
  where(l) x = y
end subroutine

subroutine test_intrinsic_where_2(x, y, l)
  use defined_assignments
  logical :: x(10), l(10)
  real :: y(10)
  where(l) y = y.lt.0.
end subroutine

subroutine test_scalar_func_but_not_elemental(x, y)
  interface assignment(=)
    ! scalar, but not elemental
    elemental subroutine assign_integer_to_logical(a,b)
      logical, intent(out) :: a
      integer, intent(in) :: b
    end
  end interface
  logical :: x(100)
  integer :: y(100)
  ! Scalar assignment in forall should be treated just like elemental
  ! functions.
  forall(i=1:10) x(i) = y(i)
end subroutine

subroutine test_in_forall_with_cleanup(x, y)
  use defined_assignments
  interface
    pure function returns_alloc(i)
      integer, intent(in) :: i
      real, allocatable :: returns_alloc
    end function
  end interface
  logical :: x(10)
  real :: y(10)
  forall (i=1:10) x(i) = returns_alloc(i)
end subroutine

! CHECK-LABEL: func @_QPtest_forall_array(
! CHECK-SAME:        %arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca f32
! CHECK:     %[[V_1:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_4:[0-9]+]] = fir.array_load %arg0 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.array<?x?x!fir.logical<4>>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
! CHECK:     %[[V_6:[0-9]+]] = fir.do_loop %arg2 = %[[V_2]] to %[[V_3]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_4]]) -> (!fir.array<?x?x!fir.logical<4>>) {
! CHECK:       %[[V_7:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_7]] to %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[C_1_0:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_8:[0-9]+]]:3 = fir.box_dims %arg0, %[[C_1_0]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
! CHECK:       %[[C_1_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_9:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_10:[0-9]+]] = fir.convert %[[V_9:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_11:[0-9]+]] = fir.convert %[[V_10:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_12:[0-9]+]] = arith.subi %[[V_11]], %[[C_1_1]] : index
! CHECK:       %[[C_1_i64:[-0-9a-z_]+]] = arith.constant 1 : i64
! CHECK:       %[[V_13:[0-9]+]] = fir.convert %[[C_1_i64]] : (i64) -> index
! CHECK:       %[[V_14:[0-9]+]] = arith.addi %[[C_1_1]], %[[V_8]]#1 : index
! CHECK:       %[[V_15:[0-9]+]] = arith.subi %[[V_14]], %[[C_1_1]] : index
! CHECK:       %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:       %[[V_16:[0-9]+]] = arith.subi %[[V_15]], %[[C_1_1]] : index
! CHECK:       %[[V_17:[0-9]+]] = arith.addi %[[V_16]], %[[V_13:[0-9]+]] : index
! CHECK:       %[[V_18:[0-9]+]] = arith.divsi %[[V_17]], %[[V_13:[0-9]+]] : index
! CHECK:       %[[V_19:[0-9]+]] = arith.cmpi sgt, %[[V_18]], %[[C_0]] : index
! CHECK:       %[[V_20:[0-9]+]] = arith.select %[[V_19]], %[[V_18]], %[[C_0]] : index
! CHECK:       %[[C_1_2:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_21:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_22:[0-9]+]] = fir.convert %[[V_21:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_23:[0-9]+]] = fir.convert %[[V_22:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_24:[0-9]+]] = arith.subi %[[V_23]], %[[C_1_2]] : index
! CHECK:       %[[C_1_i64_3:[-0-9a-z_]+]] = arith.constant 1 : i64
! CHECK:       %[[V_25:[0-9]+]] = fir.convert %[[C_1_i64_3]] : (i64) -> index
! CHECK:       %[[C_1_4:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[C_0_5:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:       %[[V_26:[0-9]+]] = arith.subi %[[V_20]], %[[C_1_4]] : index
! CHECK:       %[[V_27:[0-9]+]] = fir.do_loop %arg4 = %[[C_0_5]] to %[[V_26]] step %[[C_1_4]] unordered iter_args(%arg5 = %arg3) -> (!fir.array<?x?x!fir.logical<4>>) {
! CHECK:         %[[V_28:[0-9]+]] = arith.subi %[[C_1_2]], %[[C_1_2]] : index
! CHECK:         %[[V_29:[0-9]+]] = arith.muli %arg4, %[[V_25:[0-9]+]] : index
! CHECK:         %[[V_30:[0-9]+]] = arith.addi %[[V_28]], %[[V_29:[0-9]+]] : index
! CHECK:         %[[V_31:[0-9]+]] = fir.array_fetch %[[V_5]], %[[V_24]], %[[V_30:[0-9]+]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:         %[[V_32:[0-9]+]] = fir.no_reassoc %[[V_31:[0-9]+]] : f32
! CHECK:         %[[V_33:[0-9]+]] = arith.subi %[[C_1_1]], %[[C_1_1]] : index
! CHECK:         %[[V_34:[0-9]+]] = arith.muli %arg4, %[[V_13:[0-9]+]] : index
! CHECK:         %[[V_35:[0-9]+]] = arith.addi %[[V_33]], %[[V_34:[0-9]+]] : index
! CHECK:         %[[V_36:[0-9]+]]:2 = fir.array_modify %arg5, %[[V_12]], %[[V_35:[0-9]+]] : (!fir.array<?x?x!fir.logical<4>>, index, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<?x?x!fir.logical<4>>)
! CHECK:         fir.store %[[V_32]] to %[[V_0:[0-9]+]] : !fir.ref<f32>
! CHECK:         fir.call @_QPassign_real_to_logical(%[[V_36]]#0, %[[V_0]]) fastmath<contract> : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:         fir.result %[[V_36]]#1 : !fir.array<?x?x!fir.logical<4>>
! CHECK:       }
! CHECK:       fir.result %[[V_27:[0-9]+]] : !fir.array<?x?x!fir.logical<4>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_4]], %[[V_6]] to %arg0 : !fir.array<?x?x!fir.logical<4>>, !fir.array<?x?x!fir.logical<4>>, !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK:     return
! CHECK:   }

subroutine test_forall_array(x, y)
  use defined_assignments
  logical :: x(:, :)
  real :: y(:, :)
  forall (i=1:10) x(i, :) = y(i, :)
end subroutine

! CHECK-LABEL: func @_QPfrom_char_forall(
! CHECK-SAME:       %arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"}, %arg1: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_1:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_3:[0-9]+]] = fir.array_load %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:     %[[V_4:[0-9]+]] = fir.array_load %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:     %[[V_5:[0-9]+]] = fir.do_loop %arg2 = %[[V_1]] to %[[V_2]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_3]]) -> (!fir.array<?xi32>) {
! CHECK:       %[[V_6:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_6]] to %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[C_1_0:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_7:[0-9]+]] = fir.load %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_8:[0-9]+]] = fir.convert %[[V_7:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_9:[0-9]+]] = fir.convert %[[V_8:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_10:[0-9]+]] = arith.subi %[[V_9]], %[[C_1_0]] : index
! CHECK:       %[[V_11:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:       %[[C_1_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_12:[0-9]+]] = arith.divsi %[[V_11]], %[[C_1_1]] : index
! CHECK:       %[[V_13:[0-9]+]] = fir.array_access %[[V_4]], %[[V_10]] typeparams %[[V_12:[0-9]+]] : (!fir.array<?x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:       %[[V_14:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:       %[[V_15:[0-9]+]] = fir.no_reassoc %[[V_13:[0-9]+]] : !fir.ref<!fir.char<1,?>>
! CHECK:       %[[C_1_2:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_16:[0-9]+]] = fir.load %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_17:[0-9]+]] = fir.convert %[[V_16:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_18:[0-9]+]] = fir.convert %[[V_17:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_19:[0-9]+]] = arith.subi %[[V_18]], %[[C_1_2]] : index
! CHECK:       %[[V_20:[0-9]+]]:2 = fir.array_modify %arg3, %[[V_19:[0-9]+]] : (!fir.array<?xi32>, index) -> (!fir.ref<i32>, !fir.array<?xi32>)
! CHECK:       %[[V_21:[0-9]+]] = fir.emboxchar %[[V_15]], %[[V_14:[0-9]+]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:       fir.call @_QPsfrom_char(%[[V_20]]#0, %[[V_21]]) fastmath<contract> : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:       fir.result %[[V_20]]#1 : !fir.array<?xi32>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_3]], %[[V_5]] to %arg0 : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPto_char_forall(
! CHECK-SAME:        %arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"}, %arg1: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca i32
! CHECK:     %[[V_1:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_4:[0-9]+]] = fir.array_load %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:     %[[V_6:[0-9]+]] = fir.do_loop %arg2 = %[[V_2]] to %[[V_3]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_4]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:       %[[V_7:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_7]] to %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[C_1_0:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_8:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_9:[0-9]+]] = fir.convert %[[V_8:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_10:[0-9]+]] = fir.convert %[[V_9:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_11:[0-9]+]] = arith.subi %[[V_10]], %[[C_1_0]] : index
! CHECK:       %[[V_12:[0-9]+]] = fir.array_fetch %[[V_5]], %[[V_11:[0-9]+]] : (!fir.array<?xi32>, index) -> i32
! CHECK:       %[[V_13:[0-9]+]] = fir.no_reassoc %[[V_12:[0-9]+]] : i32
! CHECK:       %[[C_1_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_14:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_15:[0-9]+]] = fir.convert %[[V_14:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_16:[0-9]+]] = fir.convert %[[V_15:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_17:[0-9]+]] = arith.subi %[[V_16]], %[[C_1_1]] : index
! CHECK:       %[[V_18:[0-9]+]]:2 = fir.array_modify %arg3, %[[V_17:[0-9]+]] : (!fir.array<?x!fir.char<1,?>>, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>)
! CHECK:       %[[V_19:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:       %[[V_20:[0-9]+]] = fir.emboxchar %[[V_18]]#0, %[[V_19:[0-9]+]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:       fir.store %[[V_13]] to %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       fir.call @_QPsto_char(%[[V_20]], %[[V_0]]) fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:       fir.result %[[V_18]]#1 : !fir.array<?x!fir.char<1,?>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_4]], %[[V_6]] to %arg1 : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:     return
! CHECK:   }

subroutine from_char_forall(i, c)
  interface assignment(=)
    elemental subroutine sfrom_char(a,b)
      integer, intent(out) :: a
      character(*),intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  forall (j=1:10) i(j) = c(j)
end subroutine

subroutine to_char_forall(i, c)
  interface assignment(=)
    elemental subroutine sto_char(a,b)
      character(*), intent(out) :: a
      integer,intent(in) :: b
    end subroutine
  end interface
  integer :: i(:)
  character(*) :: c(:)
  forall (j=1:10) c(j) = i(j)
end subroutine

! CHECK-LABEL: func @_QPfrom_char_forall_array(
! CHECK-SAME:        %arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "i"}, %arg1: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_1:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_3:[0-9]+]] = fir.array_load %arg0 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.array<?x?xi32>
! CHECK:     %[[V_4:[0-9]+]] = fir.array_load %arg1 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.array<?x?x!fir.char<1,?>>
! CHECK:     %[[V_5:[0-9]+]] = fir.do_loop %arg2 = %[[V_1]] to %[[V_2]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_3]]) -> (!fir.array<?x?xi32>) {
! CHECK:       %[[V_6:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_6]] to %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[C_1_0:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_7:[0-9]+]]:3 = fir.box_dims %arg0, %[[C_1_0]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:       %[[C_1_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_8:[0-9]+]] = fir.load %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_9:[0-9]+]] = fir.convert %[[V_8:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_10:[0-9]+]] = fir.convert %[[V_9:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_11:[0-9]+]] = arith.subi %[[V_10]], %[[C_1_1]] : index
! CHECK:       %[[C_1_i64:[-0-9a-z_]+]] = arith.constant 1 : i64
! CHECK:       %[[V_12:[0-9]+]] = fir.convert %[[C_1_i64]] : (i64) -> index
! CHECK:       %[[V_13:[0-9]+]] = arith.addi %[[C_1_1]], %[[V_7]]#1 : index
! CHECK:       %[[V_14:[0-9]+]] = arith.subi %[[V_13]], %[[C_1_1]] : index
! CHECK:       %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:       %[[V_15:[0-9]+]] = arith.subi %[[V_14]], %[[C_1_1]] : index
! CHECK:       %[[V_16:[0-9]+]] = arith.addi %[[V_15]], %[[V_12:[0-9]+]] : index
! CHECK:       %[[V_17:[0-9]+]] = arith.divsi %[[V_16]], %[[V_12:[0-9]+]] : index
! CHECK:       %[[V_18:[0-9]+]] = arith.cmpi sgt, %[[V_17]], %[[C_0]] : index
! CHECK:       %[[V_19:[0-9]+]] = arith.select %[[V_18]], %[[V_17]], %[[C_0]] : index
! CHECK:       %[[C_1_2:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_20:[0-9]+]] = fir.load %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_21:[0-9]+]] = fir.convert %[[V_20:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_22:[0-9]+]] = fir.convert %[[V_21:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_23:[0-9]+]] = arith.subi %[[V_22]], %[[C_1_2]] : index
! CHECK:       %[[C_1_i64_3:[-0-9a-z_]+]] = arith.constant 1 : i64
! CHECK:       %[[V_24:[0-9]+]] = fir.convert %[[C_1_i64_3]] : (i64) -> index
! CHECK:       %[[C_1_4:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[C_0_5:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:       %[[V_25:[0-9]+]] = arith.subi %[[V_19]], %[[C_1_4]] : index
! CHECK:       %[[V_26:[0-9]+]] = fir.do_loop %arg4 = %[[C_0_5]] to %[[V_25]] step %[[C_1_4]] unordered iter_args(%arg5 = %arg3) -> (!fir.array<?x?xi32>) {
! CHECK:         %[[V_27:[0-9]+]] = arith.subi %[[C_1_2]], %[[C_1_2]] : index
! CHECK:         %[[V_28:[0-9]+]] = arith.muli %arg4, %[[V_24:[0-9]+]] : index
! CHECK:         %[[V_29:[0-9]+]] = arith.addi %[[V_27]], %[[V_28:[0-9]+]] : index
! CHECK:         %[[V_30:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:         %[[C_1_6:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:         %[[V_31:[0-9]+]] = arith.divsi %[[V_30]], %[[C_1_6]] : index
! CHECK:         %[[V_32:[0-9]+]] = fir.array_access %[[V_4]], %[[V_23]], %[[V_29]] typeparams %[[V_31:[0-9]+]] : (!fir.array<?x?x!fir.char<1,?>>, index, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[V_33:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:         %[[V_34:[0-9]+]] = fir.no_reassoc %[[V_32:[0-9]+]] : !fir.ref<!fir.char<1,?>>
! CHECK:         %[[V_35:[0-9]+]] = arith.subi %[[C_1_1]], %[[C_1_1]] : index
! CHECK:         %[[V_36:[0-9]+]] = arith.muli %arg4, %[[V_12:[0-9]+]] : index
! CHECK:         %[[V_37:[0-9]+]] = arith.addi %[[V_35]], %[[V_36:[0-9]+]] : index
! CHECK:         %[[V_38:[0-9]+]]:2 = fir.array_modify %arg5, %[[V_11]], %[[V_37:[0-9]+]] : (!fir.array<?x?xi32>, index, index) -> (!fir.ref<i32>, !fir.array<?x?xi32>)
! CHECK:         %[[V_39:[0-9]+]] = fir.emboxchar %[[V_34]], %[[V_33:[0-9]+]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPsfrom_char(%[[V_38]]#0, %[[V_39]]) fastmath<contract> : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:         fir.result %[[V_38]]#1 : !fir.array<?x?xi32>
! CHECK:       }
! CHECK:       fir.result %[[V_26:[0-9]+]] : !fir.array<?x?xi32>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_3]], %[[V_5]] to %arg0 : !fir.array<?x?xi32>, !fir.array<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:     return
! CHECK:   }

! CHECK-LABEL: func @_QPto_char_forall_array(
! CHECK-SAME:      %arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "i"}, %arg1: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca i32
! CHECK:     %[[V_1:[0-9]+]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:     %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[C_1_i32]] : (i32) -> index
! CHECK:     %[[C_10_i32:[-0-9a-z_]+]] = arith.constant 10 : i32
! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[C_10_i32]] : (i32) -> index
! CHECK:     %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:     %[[V_4:[0-9]+]] = fir.array_load %arg1 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.array<?x?x!fir.char<1,?>>
! CHECK:     %[[V_5:[0-9]+]] = fir.array_load %arg0 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.array<?x?xi32>
! CHECK:     %[[V_6:[0-9]+]] = fir.do_loop %arg2 = %[[V_2]] to %[[V_3]] step %[[C_1]] unordered iter_args(%arg3 = %[[V_4]]) -> (!fir.array<?x?x!fir.char<1,?>>) {
! CHECK:       %[[V_7:[0-9]+]] = fir.convert %arg2 : (index) -> i32
! CHECK:       fir.store %[[V_7]] to %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[C_1_0:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_8:[0-9]+]]:3 = fir.box_dims %arg1, %[[C_1_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:       %[[C_1_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_9:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_10:[0-9]+]] = fir.convert %[[V_9:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_11:[0-9]+]] = fir.convert %[[V_10:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_12:[0-9]+]] = arith.subi %[[V_11]], %[[C_1_1]] : index
! CHECK:       %[[C_1_i64:[-0-9a-z_]+]] = arith.constant 1 : i64
! CHECK:       %[[V_13:[0-9]+]] = fir.convert %[[C_1_i64]] : (i64) -> index
! CHECK:       %[[V_14:[0-9]+]] = arith.addi %[[C_1_1]], %[[V_8]]#1 : index
! CHECK:       %[[V_15:[0-9]+]] = arith.subi %[[V_14]], %[[C_1_1]] : index
! CHECK:       %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:       %[[V_16:[0-9]+]] = arith.subi %[[V_15]], %[[C_1_1]] : index
! CHECK:       %[[V_17:[0-9]+]] = arith.addi %[[V_16]], %[[V_13:[0-9]+]] : index
! CHECK:       %[[V_18:[0-9]+]] = arith.divsi %[[V_17]], %[[V_13:[0-9]+]] : index
! CHECK:       %[[V_19:[0-9]+]] = arith.cmpi sgt, %[[V_18]], %[[C_0]] : index
! CHECK:       %[[V_20:[0-9]+]] = arith.select %[[V_19]], %[[V_18]], %[[C_0]] : index
! CHECK:       %[[C_1_2:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[V_21:[0-9]+]] = fir.load %[[V_1:[0-9]+]] : !fir.ref<i32>
! CHECK:       %[[V_22:[0-9]+]] = fir.convert %[[V_21:[0-9]+]] : (i32) -> i64
! CHECK:       %[[V_23:[0-9]+]] = fir.convert %[[V_22:[0-9]+]] : (i64) -> index
! CHECK:       %[[V_24:[0-9]+]] = arith.subi %[[V_23]], %[[C_1_2]] : index
! CHECK:       %[[C_1_i64_3:[-0-9a-z_]+]] = arith.constant 1 : i64
! CHECK:       %[[V_25:[0-9]+]] = fir.convert %[[C_1_i64_3]] : (i64) -> index
! CHECK:       %[[C_1_4:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK:       %[[C_0_5:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:       %[[V_26:[0-9]+]] = arith.subi %[[V_20]], %[[C_1_4]] : index
! CHECK:       %[[V_27:[0-9]+]] = fir.do_loop %arg4 = %[[C_0_5]] to %[[V_26]] step %[[C_1_4]] unordered iter_args(%arg5 = %arg3) -> (!fir.array<?x?x!fir.char<1,?>>) {
! CHECK:         %[[V_28:[0-9]+]] = arith.subi %[[C_1_2]], %[[C_1_2]] : index
! CHECK:         %[[V_29:[0-9]+]] = arith.muli %arg4, %[[V_25:[0-9]+]] : index
! CHECK:         %[[V_30:[0-9]+]] = arith.addi %[[V_28]], %[[V_29:[0-9]+]] : index
! CHECK:         %[[V_31:[0-9]+]] = fir.array_fetch %[[V_5]], %[[V_24]], %[[V_30:[0-9]+]] : (!fir.array<?x?xi32>, index, index) -> i32
! CHECK:         %[[V_32:[0-9]+]] = fir.no_reassoc %[[V_31:[0-9]+]] : i32
! CHECK:         %[[V_33:[0-9]+]] = arith.subi %[[C_1_1]], %[[C_1_1]] : index
! CHECK:         %[[V_34:[0-9]+]] = arith.muli %arg4, %[[V_13:[0-9]+]] : index
! CHECK:         %[[V_35:[0-9]+]] = arith.addi %[[V_33]], %[[V_34:[0-9]+]] : index
! CHECK:         %[[V_36:[0-9]+]]:2 = fir.array_modify %arg5, %[[V_12]], %[[V_35:[0-9]+]] : (!fir.array<?x?x!fir.char<1,?>>, index, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x?x!fir.char<1,?>>)
! CHECK:         %[[V_37:[0-9]+]] = fir.box_elesize %arg1 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:         %[[V_38:[0-9]+]] = fir.emboxchar %[[V_36]]#0, %[[V_37:[0-9]+]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.store %[[V_32]] to %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK:         fir.call @_QPsto_char(%[[V_38]], %[[V_0]]) fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:         fir.result %[[V_36]]#1 : !fir.array<?x?x!fir.char<1,?>>
! CHECK:       }
! CHECK:       fir.result %[[V_27:[0-9]+]] : !fir.array<?x?x!fir.char<1,?>>
! CHECK:     }
! CHECK:     fir.array_merge_store %[[V_4]], %[[V_6]] to %arg1 : !fir.array<?x?x!fir.char<1,?>>, !fir.array<?x?x!fir.char<1,?>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:     return
! CHECK:   }

subroutine from_char_forall_array(i, c)
  interface assignment(=)
    elemental subroutine sfrom_char(a,b)
      integer, intent(out) :: a
      character(*),intent(in) :: b
    end subroutine
  end interface
  integer :: i(:, :)
  character(*) :: c(:, :)
  forall (j=1:10) i(j, :) = c(j, :)
end subroutine

subroutine to_char_forall_array(i, c)
  interface assignment(=)
    elemental subroutine sto_char(a,b)
      character(*), intent(out) :: a
      integer,intent(in) :: b
    end subroutine
  end interface
  integer :: i(:, :)
  character(*) :: c(:, :)
  forall (j=1:10) c(j, :) = i(j, :)
end subroutine

! TODO: test array user defined assignment inside FORALL.
subroutine test_todo(x, y)
  interface assignment(=)
    ! User assignment is not elemental, it takes array arguments.
    pure subroutine assign_array(a,b)
      logical, intent(out) :: a(:)
      integer, intent(in) :: b(:)
    end
  end interface
  logical :: x(10, 10)
  integer :: y(10, 10)
!  forall(i=1:10) x(i, :) = y(i, :)
end subroutine
