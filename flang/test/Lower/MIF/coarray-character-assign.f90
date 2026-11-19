! Test that assigning a character value of a different length to a
! nonallocatable coarray does not reallocate it

! RUN: %flang_fc1 -emit-fir -fcoarray %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fcoarray -fno-realloc-lhs %s -o - | FileCheck %s --check-prefix=NOREALLOC

subroutine s()
  character(4), save :: c(2)[*]
  c = "x"
end subroutine

! CHECK-LABEL: func.func @_QPs()
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<2x!fir.char<1,4>>>
! CHECK-NOT:     fir.alloca !fir.box<!fir.heap<
! CHECK:         %[[VAL_3:.*]] = fir.declare %{{.*}} typeparams %{{.*}} uniq_name("_QFsEc")
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>>
! CHECK:         %[[VAL_9:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>) -> !fir.heap<!fir.array<2x!fir.char<1,4>>>
! CHECK:         %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_6]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>, index) -> (index, index, index)
! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_11]]#1 : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_13:.*]] = fir.embox %[[VAL_9]](%[[VAL_12]]) : (!fir.heap<!fir.array<2x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,4>>>
! CHECK:         fir.store %[[VAL_13]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<2x!fir.char<1,4>>>>
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<2x!fir.char<1,4>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAssign(%[[VAL_15]], {{.*}})

! An allocatable coarray must keep the reallocation semantics: the descriptor
! given to the runtime is still the heap descriptor of the variable.
subroutine s_alloc()
  character(:), allocatable :: c(:)[:]
  allocate(character(4) :: c(2)[*])
  c = "x"
end subroutine

! CHECK-LABEL: func.func @_QPs_alloc()
! CHECK:         %[[VAL_7:.*]] = fir.declare %{{.*}} fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, corank:1>>
! CHECK:         mif.alloc_coarray %[[VAL_7]]
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, corank:1>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAssign(%[[VAL_19]], {{.*}})

! Allocatable coarray with a whole array assignment. This used to hit an
! assertion in fir::BoxValue because the LHS extended value is a data address,
! not a descriptor.
subroutine s_int_alloc()
  integer, allocatable :: c(:)[:]
  allocate(c(2)[*])
  c = [10, 20]
end subroutine

! CHECK-LABEL: func.func @_QPs_int_alloc()
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>
! CHECK:         %[[VAL_8:.*]] = fir.declare %[[VAL_4]] uniq_name("_QFs_int_allocEc") fortran_attrs<allocatable>
! CHECK:         fir.store %{{.*}} to %[[VAL_8]]
! CHECK:         %[[VAL_23:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
! CHECK:         %[[VAL_24:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_23]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>, index) -> (index, index, index)
! CHECK:         %[[VAL_28:.*]] = fir.shape_shift %[[VAL_25]]#0, %[[VAL_25]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_29:.*]] = fir.embox %[[VAL_24]](%[[VAL_28]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         fir.store %[[VAL_29]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:         %[[VAL_31:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAssignSimple(%[[VAL_31]], {{.*}})

! CONTIGUOUS nonallocatable coarray: same extended value (data address).
subroutine s_contiguous()
  character(4), contiguous, save :: c(2)[*]
  c = "x"
end subroutine

! CHECK-LABEL: func.func @_QPs_contiguous()
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<2x!fir.char<1,4>>>
! CHECK-NOT:     fir.alloca !fir.box<!fir.heap<
! CHECK:         %[[VAL_6:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>>
! CHECK:         %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>) -> !fir.heap<!fir.array<2x!fir.char<1,4>>>
! CHECK:         %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_6]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>, index) -> (index, index, index)
! CHECK:         %[[VAL_10:.*]] = fir.shape_shift %[[VAL_8]]#0, %[[VAL_8]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_7]](%[[VAL_10]]) : (!fir.heap<!fir.array<2x!fir.char<1,4>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<2x!fir.char<1,4>>>
! CHECK:         fir.store %[[VAL_11]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<2x!fir.char<1,4>>>>
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<2x!fir.char<1,4>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         fir.call @_FortranAAssign(%[[VAL_13]], {{.*}})

! Allocatable coarray with -fno-realloc-lhs.
subroutine s_alloc_norealloc()
  character(4), allocatable :: c(:)[:]
  allocate(c(2)[*])
  c = "x"
end subroutine

! NOREALLOC-LABEL: func.func @_QPs_alloc_norealloc()
! NOREALLOC:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<?x!fir.char<1,4>>>
! NOREALLOC:         %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, corank:1>
! NOREALLOC:         %[[VAL_8:.*]] = fir.declare %[[VAL_4]] typeparams %{{.*}} uniq_name("_QFs_alloc_noreallocEc") fortran_attrs<allocatable>
! NOREALLOC:         mif.alloc_coarray %[[VAL_8]]
! NOREALLOC:         %[[VAL_17:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, corank:1>>
! NOREALLOC:         %[[VAL_18:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, corank:1>) -> !fir.heap<!fir.array<?x!fir.char<1,4>>>
! NOREALLOC:         %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_17]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, corank:1>, index) -> (index, index, index)
! NOREALLOC:         %[[VAL_21:.*]] = fir.shape_shift %[[VAL_19]]#0, %[[VAL_19]]#1 : (index, index) -> !fir.shapeshift<1>
! NOREALLOC:         %[[VAL_22:.*]] = fir.embox %[[VAL_18]](%[[VAL_21]]) : (!fir.heap<!fir.array<?x!fir.char<1,4>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.char<1,4>>>
! NOREALLOC:         fir.store %[[VAL_22]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?x!fir.char<1,4>>>>
! NOREALLOC:         %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<?x!fir.char<1,4>>>>) -> !fir.ref<!fir.box<none>>
! NOREALLOC:         fir.call @_FortranAAssign(%[[VAL_24]], {{.*}})
