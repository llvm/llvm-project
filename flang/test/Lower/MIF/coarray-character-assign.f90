! Test that assigning a character value of a different length to a
! nonallocatable coarray does not reallocate it

! RUN: %flang_fc1 -emit-fir -fcoarray %s -o - | FileCheck %s

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
