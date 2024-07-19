! Test lowering of COPYPRIVATE with allocatable/pointer variables.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: func private @_copy_box_ptr_i32(
!CHECK-SAME:                  %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>,
!CHECK-SAME:                  %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>) {
!CHECK-NEXT:    %[[DST:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<pointer>,
!CHECK-SAME:      uniq_name = "_copy_box_ptr_i32_dst"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) ->
!CHECK-SAME:      (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-NEXT:    %[[SRC:.*]]:2 = hlfir.declare %[[ARG1]] {fortran_attrs = #fir.var_attrs<pointer>,
!CHECK-SAME:      uniq_name = "_copy_box_ptr_i32_src"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) ->
!CHECK-SAME:      (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-NEXT:    %[[SRC_VAL:.*]] = fir.load %[[SRC]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-NEXT:    fir.store %[[SRC_VAL]] to %[[DST]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-NEXT:    return
!CHECK-NEXT:  }

!CHECK-LABEL: func private @_copy_box_heap_Uxi32(
!CHECK-SAME:        %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>,
!CHECK-SAME:        %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
!CHECK-NEXT:    %[[DST:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<allocatable>,
!CHECK-SAME:      uniq_name = "_copy_box_heap_Uxi32_dst"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) ->
!CHECK-SAME:      (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK-NEXT:    %[[SRC:.*]]:2 = hlfir.declare %[[ARG1]] {fortran_attrs = #fir.var_attrs<allocatable>,
!CHECK-SAME:      uniq_name = "_copy_box_heap_Uxi32_src"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) ->
!CHECK-SAME:      (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK-NEXT:    %[[DST_BOX:.*]] = fir.load %[[DST]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:         fir.if %{{.*}} {
!CHECK-NEXT:      %[[SRC_BOX:.*]] = fir.load %[[SRC]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK-NEXT:      hlfir.assign %[[SRC_BOX]] to %[[DST_BOX]] temporary_lhs : !fir.box<!fir.heap<!fir.array<?xi32>>>,
!CHECK-SAME:        !fir.box<!fir.heap<!fir.array<?xi32>>>
!CHECK-NEXT:    }
!CHECK-NEXT:    return
!CHECK-NEXT:  }

!CHECK-LABEL: func @_QPtest_alloc_ptr
!CHECK:         omp.parallel {
!CHECK:           %[[A:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>,
!CHECK-SAME:        uniq_name = "_QFtest_alloc_ptrEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) ->
!CHECK-SAME:        (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK:           %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>,
!CHECK-SAME:        uniq_name = "_QFtest_alloc_ptrEp"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) ->
!CHECK-SAME:        (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:           omp.single copyprivate(
!CHECK-SAME:        %[[A]]#0 -> @_copy_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>,
!CHECK-SAME:        %[[P]]#0 -> @_copy_box_ptr_i32 : !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHEK:          }
subroutine test_alloc_ptr()
  integer, allocatable :: a(:)
  integer, pointer :: p

  !$omp parallel private(a, p)
  !$omp single
  !$omp end single copyprivate(a, p)
  !$omp end parallel
end subroutine
