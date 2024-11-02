! Test lower of elemental user defined assignments
! RUN: bbc -emit-fir %s -o - | FileCheck %s

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
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_3:.*]] = fir.array_load %[[VAL_0]](%[[VAL_2]]) : (!fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>, !fir.shape<1>) -> !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 100 : i64
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant -1 : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.slice %[[VAL_5]], %[[VAL_9]], %[[VAL_7]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_10]]) {{\[}}%[[VAL_11]]] : (!fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_15:.*]] = arith.subi %[[VAL_1]], %[[VAL_13]] : index
! CHECK:         %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_14]] to %[[VAL_15]] step %[[VAL_13]] unordered iter_args(%[[VAL_18:.*]] = %[[VAL_3]]) -> (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>) {
! CHECK:           %[[VAL_19:.*]] = fir.array_access %[[VAL_12]], %[[VAL_17]] : (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, index) -> !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:           %[[VAL_20:.*]]:2 = fir.array_modify %[[VAL_18]], %[[VAL_17]] : (!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, index) -> (!fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>)
! CHECK:           fir.call @_QPassign_t(%[[VAL_20]]#0, %[[VAL_19]]) : (!fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.ref<!fir.type<_QMdefined_assignmentsTt{i:i32}>>) -> ()
! CHECK:           fir.result %[[VAL_20]]#1 : !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_3]], %[[VAL_21:.*]] to %[[VAL_0]] : !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>, !fir.ref<!fir.array<100x!fir.type<_QMdefined_assignmentsTt{i:i32}>>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPtest_intrinsic(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.logical<4>
! CHECK:         %[[VAL_2:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]](%[[VAL_3]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_5:.*]] = arith.constant 100 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant -1 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.slice %[[VAL_6]], %[[VAL_10]], %[[VAL_8]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) {{\[}}%[[VAL_12]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_14:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_17:.*]] = arith.subi %[[VAL_2]], %[[VAL_15]] : index
! CHECK:         %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_16]] to %[[VAL_17]] step %[[VAL_15]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_4]]) -> (!fir.array<100xf32>) {
! CHECK:           %[[VAL_21:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_19]] : (!fir.array<100xf32>, index) -> f32
! CHECK:           %[[VAL_22:.*]] = arith.cmpf olt, %[[VAL_21]], %[[VAL_14]] : f32
! CHECK:           %[[VAL_23:.*]]:2 = fir.array_modify %[[VAL_20]], %[[VAL_19]] : (!fir.array<100xf32>, index) -> (!fir.ref<f32>, !fir.array<100xf32>)
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_22]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[VAL_24]] to %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:           fir.call @_QPassign_logical_to_real(%[[VAL_23]]#0, %[[VAL_1]]) : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:           fir.result %[[VAL_23]]#1 : !fir.array<100xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_25:.*]] to %[[VAL_0]] : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPtest_intrinsic_2(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca f32
! CHECK:         %[[VAL_3:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_0]](%[[VAL_5]]) : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<100x!fir.logical<4>>
! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_1]](%[[VAL_7]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_3]], %[[VAL_9]] : index
! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_9]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_6]]) -> (!fir.array<100x!fir.logical<4>>) {
! CHECK:           %[[VAL_15:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_13]] : (!fir.array<100xf32>, index) -> f32
! CHECK:           %[[VAL_16:.*]]:2 = fir.array_modify %[[VAL_14]], %[[VAL_13]] : (!fir.array<100x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<100x!fir.logical<4>>)
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK:           fir.call @_QPassign_real_to_logical(%[[VAL_16]]#0, %[[VAL_2]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:           fir.result %[[VAL_16]]#1 : !fir.array<100x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_6]], %[[VAL_17:.*]] to %[[VAL_0]] : !fir.array<100x!fir.logical<4>>, !fir.array<100x!fir.logical<4>>, !fir.ref<!fir.array<100x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPfrom_char(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_6:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = arith.divsi %[[VAL_6]], %[[VAL_7]] : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_9]] : index
! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_9]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_4]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_15:.*]] = fir.array_access %[[VAL_5]], %[[VAL_13]] typeparams %[[VAL_8]] : (!fir.array<?x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_16:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_17:.*]]:2 = fir.array_modify %[[VAL_14]], %[[VAL_13]] : (!fir.array<?xi32>, index) -> (!fir.ref<i32>, !fir.array<?xi32>)
! CHECK:           %[[VAL_18:.*]] = fir.emboxchar %[[VAL_15]], %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPsfrom_char(%[[VAL_17]]#0, %[[VAL_18]]) : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:           fir.result %[[VAL_17]]#1 : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_19:.*]] to %[[VAL_0]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPto_char(
! CHECK-SAME:                   %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_9:.*]] = arith.subi %[[VAL_4]]#1, %[[VAL_7]] : index
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_5]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:           %[[VAL_13:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_11]] : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_14:.*]]:2 = fir.array_modify %[[VAL_12]], %[[VAL_11]] : (!fir.array<?x!fir.char<1,?>>, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>)
! CHECK:           %[[VAL_15:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_16:.*]] = fir.emboxchar %[[VAL_14]]#0, %[[VAL_15]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           fir.call @_QPsto_char(%[[VAL_16]], %[[VAL_2]]) : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:           fir.result %[[VAL_14]]#1 : !fir.array<?x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_5]], %[[VAL_17:.*]] to %[[VAL_1]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         return
! CHECK:       }

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
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca f32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_12]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_19]] : index
! CHECK:           %[[VAL_24:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_23]] : (!fir.array<10xf32>, index) -> f32
! CHECK:           %[[VAL_25:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
! CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_28]], %[[VAL_25]] : index
! CHECK:           %[[VAL_30:.*]]:2 = fir.array_modify %[[VAL_17]], %[[VAL_29]] : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:           fir.store %[[VAL_24]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK:           fir.call @_QPassign_real_to_logical(%[[VAL_30]]#0, %[[VAL_2]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:           fir.result %[[VAL_30]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_31:.*]] to %[[VAL_0]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPtest_in_forall_2(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.logical<4>
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_1]](%[[VAL_10]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_9]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (!fir.array<10xf32>) {
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_16]] : index
! CHECK-DAG:       %[[VAL_21:.*]] = arith.constant 0.000000e+00 : f32
! CHECK-DAG:       %[[VAL_22:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_20]] : (!fir.array<10xf32>, index) -> f32
! CHECK:           %[[VAL_23:.*]] = arith.cmpf olt, %[[VAL_22]], %[[VAL_21]] : f32
! CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           %[[VAL_28:.*]] = arith.subi %[[VAL_27]], %[[VAL_24]] : index
! CHECK:           %[[VAL_29:.*]]:2 = fir.array_modify %[[VAL_14]], %[[VAL_28]] : (!fir.array<10xf32>, index) -> (!fir.ref<f32>, !fir.array<10xf32>)
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_23]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[VAL_30]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:           fir.call @_QPassign_logical_to_real(%[[VAL_29]]#0, %[[VAL_2]]) : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:           fir.result %[[VAL_29]]#1 : !fir.array<10xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_31:.*]] to %[[VAL_1]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPtest_intrinsic_where_1(
! CHECK-SAME:             %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>{{.*}}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca f32
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_2]](%[[VAL_9]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_11:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_11]](%[[VAL_12]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_16:.*]] = arith.subi %[[VAL_8]], %[[VAL_14]] : index
! CHECK:         %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_14]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_13]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_20:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_18]] : (!fir.array<10x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:           %[[VAL_21:.*]] = fir.array_update %[[VAL_19]], %[[VAL_20]], %[[VAL_18]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
! CHECK:           fir.result %[[VAL_21]] : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_13]], %[[VAL_22:.*]] to %[[VAL_11]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:         %[[VAL_23:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_24:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_25:.*]] = fir.array_load %[[VAL_0]](%[[VAL_24]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_26:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_27:.*]] = fir.array_load %[[VAL_1]](%[[VAL_26]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_28:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_30:.*]] = arith.subi %[[VAL_5]], %[[VAL_28]] : index
! CHECK:         %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_25]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_34:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_35:.*]] = arith.addi %[[VAL_32]], %[[VAL_34]] : index
! CHECK:           %[[VAL_36:.*]] = fir.array_coor %[[VAL_11]](%[[VAL_23]]) %[[VAL_35]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_36]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.logical<4>) -> i1
! CHECK:           %[[VAL_39:.*]] = fir.if %[[VAL_38]] -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:             %[[VAL_40:.*]] = fir.array_fetch %[[VAL_27]], %[[VAL_32]] : (!fir.array<10xf32>, index) -> f32
! CHECK:             %[[VAL_41:.*]]:2 = fir.array_modify %[[VAL_33]], %[[VAL_32]] : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:             fir.store %[[VAL_40]] to %[[VAL_3]] : !fir.ref<f32>
! CHECK:             fir.call @_QPassign_real_to_logical(%[[VAL_41]]#0, %[[VAL_3]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:             fir.result %[[VAL_41]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_33]] : !fir.array<10x!fir.logical<4>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_42:.*]] : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_25]], %[[VAL_43:.*]] to %[[VAL_0]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:         fir.freemem %[[VAL_11]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPtest_intrinsic_where_2(
! CHECK-SAME:           %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>{{.*}}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.logical<4>
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_2]](%[[VAL_8]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_10:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_10]](%[[VAL_11]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_15:.*]] = arith.subi %[[VAL_7]], %[[VAL_13]] : index
! CHECK:         %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_14]] to %[[VAL_15]] step %[[VAL_13]] unordered iter_args(%[[VAL_18:.*]] = %[[VAL_12]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_19:.*]] = fir.array_fetch %[[VAL_9]], %[[VAL_17]] : (!fir.array<10x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:           %[[VAL_20:.*]] = fir.array_update %[[VAL_18]], %[[VAL_19]], %[[VAL_17]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
! CHECK:           fir.result %[[VAL_20]] : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_21:.*]] to %[[VAL_10]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_23:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_24:.*]] = fir.array_load %[[VAL_1]](%[[VAL_23]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_25:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_26:.*]] = fir.array_load %[[VAL_1]](%[[VAL_25]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_27:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:         %[[VAL_28:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_30:.*]] = arith.subi %[[VAL_5]], %[[VAL_28]] : index
! CHECK:         %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_24]]) -> (!fir.array<10xf32>) {
! CHECK:           %[[VAL_34:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_35:.*]] = arith.addi %[[VAL_32]], %[[VAL_34]] : index
! CHECK:           %[[VAL_36:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_22]]) %[[VAL_35]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_36]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.logical<4>) -> i1
! CHECK:           %[[VAL_39:.*]] = fir.if %[[VAL_38]] -> (!fir.array<10xf32>) {
! CHECK:             %[[VAL_40:.*]] = fir.array_fetch %[[VAL_26]], %[[VAL_32]] : (!fir.array<10xf32>, index) -> f32
! CHECK:             %[[VAL_41:.*]] = arith.cmpf olt, %[[VAL_40]], %[[VAL_27]] : f32
! CHECK:             %[[VAL_42:.*]]:2 = fir.array_modify %[[VAL_33]], %[[VAL_32]] : (!fir.array<10xf32>, index) -> (!fir.ref<f32>, !fir.array<10xf32>)
! CHECK:             %[[VAL_43:.*]] = fir.convert %[[VAL_41]] : (i1) -> !fir.logical<4>
! CHECK:             fir.store %[[VAL_43]] to %[[VAL_3]] : !fir.ref<!fir.logical<4>>
! CHECK:             fir.call @_QPassign_logical_to_real(%[[VAL_42]]#0, %[[VAL_3]]) : (!fir.ref<f32>, !fir.ref<!fir.logical<4>>) -> ()
! CHECK:             fir.result %[[VAL_42]]#1 : !fir.array<10xf32>
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_33]] : !fir.array<10xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_44:.*]] : !fir.array<10xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_24]], %[[VAL_45:.*]] to %[[VAL_1]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:         fir.freemem %[[VAL_10]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPtest_scalar_func_but_not_elemental(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<100x!fir.logical<4>>
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_12]]) -> (!fir.array<100x!fir.logical<4>>) {
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_19]] : index
! CHECK:           %[[VAL_24:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_23]] : (!fir.array<100xi32>, index) -> i32
! CHECK:           %[[VAL_25:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
! CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_28]], %[[VAL_25]] : index
! CHECK:           %[[VAL_30:.*]]:2 = fir.array_modify %[[VAL_17]], %[[VAL_29]] : (!fir.array<100x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<100x!fir.logical<4>>)
! CHECK:           fir.store %[[VAL_24]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           fir.call @_QPassign_integer_to_logical(%[[VAL_30]]#0, %[[VAL_2]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<i32>) -> ()
! CHECK:           fir.result %[[VAL_30]]#1 : !fir.array<100x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_31:.*]] to %[[VAL_0]] : !fir.array<100x!fir.logical<4>>, !fir.array<100x!fir.logical<4>>, !fir.ref<!fir.array<100x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPtest_in_forall_with_cleanup(
! CHECK-SAME:       %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = ".result"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]](%[[VAL_10]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_9]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (!fir.array<10x!fir.logical<4>>) {
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.call @_QPreturns_alloc(%[[VAL_3]]) : (!fir.ref<i32>) -> !fir.box<!fir.heap<f32>>
! CHECK:           fir.save_result %[[VAL_16]] to %[[VAL_2]] : !fir.box<!fir.heap<f32>>, !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_18:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_19]] : index
! CHECK:           %[[VAL_24:.*]]:2 = fir.array_modify %[[VAL_14]], %[[VAL_23]] : (!fir.array<10x!fir.logical<4>>, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<10x!fir.logical<4>>)
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_18]] : (!fir.heap<f32>) -> !fir.ref<f32>
! CHECK:           fir.call @_QPassign_real_to_logical(%[[VAL_24]]#0, %[[VAL_25]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_27:.*]] = fir.box_addr %[[VAL_26]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (!fir.heap<f32>) -> i64
! CHECK:           %[[VAL_29:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_30:.*]] = arith.cmpi ne, %[[VAL_28]], %[[VAL_29]] : i64
! CHECK:           fir.if %[[VAL_30]] {
! CHECK:             fir.freemem %[[VAL_27]] : !fir.heap<f32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_24]]#1 : !fir.array<10x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_31:.*]] to %[[VAL_0]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }



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
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca f32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.array<?x?x!fir.logical<4>>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_9]]) -> (!fir.array<?x?x!fir.logical<4>>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_15]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_17]] : index
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_17]], %[[VAL_16]]#1 : index
! CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_24]], %[[VAL_17]] : index
! CHECK:           %[[VAL_26:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_27:.*]] = arith.subi %[[VAL_25]], %[[VAL_17]] : index
! CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_23]] : index
! CHECK:           %[[VAL_29:.*]] = arith.divsi %[[VAL_28]], %[[VAL_23]] : index
! CHECK:           %[[VAL_30:.*]] = arith.cmpi sgt, %[[VAL_29]], %[[VAL_26]] : index
! CHECK:           %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_29]], %[[VAL_26]] : index
! CHECK:           %[[VAL_32:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
! CHECK:           %[[VAL_36:.*]] = arith.subi %[[VAL_35]], %[[VAL_32]] : index
! CHECK:           %[[VAL_37:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
! CHECK:           %[[VAL_39:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_40:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_31]], %[[VAL_39]] : index
! CHECK:           %[[VAL_42:.*]] = fir.do_loop %[[VAL_43:.*]] = %[[VAL_40]] to %[[VAL_41]] step %[[VAL_39]] unordered iter_args(%[[VAL_44:.*]] = %[[VAL_13]]) -> (!fir.array<?x?x!fir.logical<4>>) {
! CHECK:             %[[VAL_45:.*]] = arith.subi %[[VAL_32]], %[[VAL_32]] : index
! CHECK:             %[[VAL_46:.*]] = arith.muli %[[VAL_43]], %[[VAL_38]] : index
! CHECK:             %[[VAL_47:.*]] = arith.addi %[[VAL_45]], %[[VAL_46]] : index
! CHECK:             %[[VAL_48:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_36]], %[[VAL_47]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:             %[[VAL_49:.*]] = arith.subi %[[VAL_17]], %[[VAL_17]] : index
! CHECK:             %[[VAL_50:.*]] = arith.muli %[[VAL_43]], %[[VAL_23]] : index
! CHECK:             %[[VAL_51:.*]] = arith.addi %[[VAL_49]], %[[VAL_50]] : index
! CHECK:             %[[VAL_52:.*]]:2 = fir.array_modify %[[VAL_44]], %[[VAL_21]], %[[VAL_51]] : (!fir.array<?x?x!fir.logical<4>>, index, index) -> (!fir.ref<!fir.logical<4>>, !fir.array<?x?x!fir.logical<4>>)
! CHECK:             fir.store %[[VAL_48]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK:             fir.call @_QPassign_real_to_logical(%[[VAL_52]]#0, %[[VAL_2]]) : (!fir.ref<!fir.logical<4>>, !fir.ref<f32>) -> ()
! CHECK:             fir.result %[[VAL_52]]#1 : !fir.array<?x?x!fir.logical<4>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_53:.*]] : !fir.array<?x?x!fir.logical<4>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_54:.*]] to %[[VAL_0]] : !fir.array<?x?x!fir.logical<4>>, !fir.array<?x?x!fir.logical<4>>, !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK:         return
! CHECK:       }

subroutine test_forall_array(x, y)
  use defined_assignments
  logical :: x(:, :)
  real :: y(:, :)
  forall (i=1:10) x(i, :) = y(i, :)
end subroutine

! CHECK-LABEL: func @_QPfrom_char_forall(
! CHECK-SAME:       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_14]] : index
! CHECK:           %[[VAL_19:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_21:.*]] = arith.divsi %[[VAL_19]], %[[VAL_20]] : index
! CHECK:           %[[VAL_22:.*]] = fir.array_access %[[VAL_9]], %[[VAL_18]] typeparams %[[VAL_21]] : (!fir.array<?x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_23:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           %[[VAL_28:.*]] = arith.subi %[[VAL_27]], %[[VAL_24]] : index
! CHECK:           %[[VAL_29:.*]]:2 = fir.array_modify %[[VAL_12]], %[[VAL_28]] : (!fir.array<?xi32>, index) -> (!fir.ref<i32>, !fir.array<?xi32>)
! CHECK:           %[[VAL_30:.*]] = fir.emboxchar %[[VAL_22]], %[[VAL_23]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPsfrom_char(%[[VAL_29]]#0, %[[VAL_30]]) : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:           fir.result %[[VAL_29]]#1 : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_31:.*]] to %[[VAL_0]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPto_char_forall(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_9]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:           %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_15]] : index
! CHECK:           %[[VAL_20:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_19]] : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
! CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_24]], %[[VAL_21]] : index
! CHECK:           %[[VAL_26:.*]]:2 = fir.array_modify %[[VAL_13]], %[[VAL_25]] : (!fir.array<?x!fir.char<1,?>>, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>)
! CHECK:           %[[VAL_27:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_28:.*]] = fir.emboxchar %[[VAL_26]]#0, %[[VAL_27]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_20]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           fir.call @_QPsto_char(%[[VAL_28]], %[[VAL_2]]) : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:           fir.result %[[VAL_26]]#1 : !fir.array<?x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_29:.*]] to %[[VAL_1]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         return
! CHECK:       }

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
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.array<?x?xi32>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.array<?x?x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<?x?xi32>) {
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_14]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_16]] : index
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_23:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]]#1 : index
! CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_16]] : index
! CHECK:           %[[VAL_25:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_26:.*]] = arith.subi %[[VAL_24]], %[[VAL_16]] : index
! CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_22]] : index
! CHECK:           %[[VAL_28:.*]] = arith.divsi %[[VAL_27]], %[[VAL_22]] : index
! CHECK:           %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_28]], %[[VAL_25]] : index
! CHECK:           %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_28]], %[[VAL_25]] : index
! CHECK:           %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> i64
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i64) -> index
! CHECK:           %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_31]] : index
! CHECK:           %[[VAL_36:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i64) -> index
! CHECK:           %[[VAL_38:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_39:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_40:.*]] = arith.subi %[[VAL_30]], %[[VAL_38]] : index
! CHECK:           %[[VAL_41:.*]] = fir.do_loop %[[VAL_42:.*]] = %[[VAL_39]] to %[[VAL_40]] step %[[VAL_38]] unordered iter_args(%[[VAL_43:.*]] = %[[VAL_12]]) -> (!fir.array<?x?xi32>) {
! CHECK:             %[[VAL_44:.*]] = arith.subi %[[VAL_31]], %[[VAL_31]] : index
! CHECK:             %[[VAL_45:.*]] = arith.muli %[[VAL_42]], %[[VAL_37]] : index
! CHECK:             %[[VAL_46:.*]] = arith.addi %[[VAL_44]], %[[VAL_45]] : index
! CHECK:             %[[VAL_47:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_48:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_49:.*]] = arith.divsi %[[VAL_47]], %[[VAL_48]] : index
! CHECK:             %[[VAL_50:.*]] = fir.array_access %[[VAL_9]], %[[VAL_35]], %[[VAL_46]] typeparams %[[VAL_49]] : (!fir.array<?x?x!fir.char<1,?>>, index, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:             %[[VAL_51:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_52:.*]] = arith.subi %[[VAL_16]], %[[VAL_16]] : index
! CHECK:             %[[VAL_53:.*]] = arith.muli %[[VAL_42]], %[[VAL_22]] : index
! CHECK:             %[[VAL_54:.*]] = arith.addi %[[VAL_52]], %[[VAL_53]] : index
! CHECK:             %[[VAL_55:.*]]:2 = fir.array_modify %[[VAL_43]], %[[VAL_20]], %[[VAL_54]] : (!fir.array<?x?xi32>, index, index) -> (!fir.ref<i32>, !fir.array<?x?xi32>)
! CHECK:             %[[VAL_56:.*]] = fir.emboxchar %[[VAL_50]], %[[VAL_51]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:             fir.call @_QPsfrom_char(%[[VAL_55]]#0, %[[VAL_56]]) : (!fir.ref<i32>, !fir.boxchar<1>) -> ()
! CHECK:             fir.result %[[VAL_55]]#1 : !fir.array<?x?xi32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_57:.*]] : !fir.array<?x?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_58:.*]] to %[[VAL_0]] : !fir.array<?x?xi32>, !fir.array<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPto_char_forall_array(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.array<?x?x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.array<?x?xi32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_9]]) -> (!fir.array<?x?x!fir.char<1,?>>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_15]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_17]] : index
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_17]], %[[VAL_16]]#1 : index
! CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_24]], %[[VAL_17]] : index
! CHECK:           %[[VAL_26:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_27:.*]] = arith.subi %[[VAL_25]], %[[VAL_17]] : index
! CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_23]] : index
! CHECK:           %[[VAL_29:.*]] = arith.divsi %[[VAL_28]], %[[VAL_23]] : index
! CHECK:           %[[VAL_30:.*]] = arith.cmpi sgt, %[[VAL_29]], %[[VAL_26]] : index
! CHECK:           %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_29]], %[[VAL_26]] : index
! CHECK:           %[[VAL_32:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
! CHECK:           %[[VAL_36:.*]] = arith.subi %[[VAL_35]], %[[VAL_32]] : index
! CHECK:           %[[VAL_37:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
! CHECK:           %[[VAL_39:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_40:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_31]], %[[VAL_39]] : index
! CHECK:           %[[VAL_42:.*]] = fir.do_loop %[[VAL_43:.*]] = %[[VAL_40]] to %[[VAL_41]] step %[[VAL_39]] unordered iter_args(%[[VAL_44:.*]] = %[[VAL_13]]) -> (!fir.array<?x?x!fir.char<1,?>>) {
! CHECK:             %[[VAL_45:.*]] = arith.subi %[[VAL_32]], %[[VAL_32]] : index
! CHECK:             %[[VAL_46:.*]] = arith.muli %[[VAL_43]], %[[VAL_38]] : index
! CHECK:             %[[VAL_47:.*]] = arith.addi %[[VAL_45]], %[[VAL_46]] : index
! CHECK:             %[[VAL_48:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_36]], %[[VAL_47]] : (!fir.array<?x?xi32>, index, index) -> i32
! CHECK:             %[[VAL_49:.*]] = arith.subi %[[VAL_17]], %[[VAL_17]] : index
! CHECK:             %[[VAL_50:.*]] = arith.muli %[[VAL_43]], %[[VAL_23]] : index
! CHECK:             %[[VAL_51:.*]] = arith.addi %[[VAL_49]], %[[VAL_50]] : index
! CHECK:             %[[VAL_52:.*]]:2 = fir.array_modify %[[VAL_44]], %[[VAL_21]], %[[VAL_51]] : (!fir.array<?x?x!fir.char<1,?>>, index, index) -> (!fir.ref<!fir.char<1,?>>, !fir.array<?x?x!fir.char<1,?>>)
! CHECK:             %[[VAL_53:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_54:.*]] = fir.emboxchar %[[VAL_52]]#0, %[[VAL_53]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:             fir.store %[[VAL_48]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:             fir.call @_QPsto_char(%[[VAL_54]], %[[VAL_2]]) : (!fir.boxchar<1>, !fir.ref<i32>) -> ()
! CHECK:             fir.result %[[VAL_52]]#1 : !fir.array<?x?x!fir.char<1,?>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_55:.*]] : !fir.array<?x?x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_56:.*]] to %[[VAL_1]] : !fir.array<?x?x!fir.char<1,?>>, !fir.array<?x?x!fir.char<1,?>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:         return
! CHECK:       }

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
