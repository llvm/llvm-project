! Test lowering of OpenMP metadirective with construct selectors.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_construct_parallel()
! CHECK:         omp.parallel
! CHECK:           omp.barrier
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_construct_parallel()
  !$omp parallel
    !$omp metadirective &
    !$omp & when(construct={parallel}: barrier) &
    !$omp & default(nothing)
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_no_match()
! CHECK:         omp.parallel
! CHECK-NOT:     omp.barrier
! CHECK:         omp.taskyield
! CHECK:         omp.terminator
! CHECK:         return
subroutine test_construct_no_match()
  !$omp parallel
    !$omp metadirective &
    !$omp & when(construct={target}: barrier) &
    !$omp & default(taskyield)
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_for(
! CHECK:         omp.wsloop
! CHECK-NOT:     omp.taskyield
! CHECK:         return
subroutine test_construct_for(n)
  integer :: i, n
  !$omp do
  do i = 1, n
    !$omp metadirective &
    !$omp & when(construct={for}: nothing) &
    !$omp & default(taskyield)
  end do
  !$omp end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_construct_parallel()
! CHECK:         omp.parallel
! CHECK:           omp.parallel
! CHECK:             omp.terminator
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_begin_construct_parallel()
  integer :: x
  x = 0
  !$omp parallel
    !$omp begin metadirective &
    !$omp & when(construct={parallel}: parallel)
    x = 1
    !$omp end metadirective
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_construct_no_match()
! CHECK:         omp.parallel
! CHECK-NOT:     omp.task
! CHECK:         omp.terminator
! CHECK:         return
subroutine test_begin_construct_no_match()
  integer :: x
  x = 0
  !$omp parallel
    !$omp begin metadirective &
    !$omp & when(construct={target}: task)
    x = 1
    !$omp end metadirective
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_construct_selected_parent()
! CHECK:         omp.target
! CHECK:           omp.parallel
! CHECK:             omp.barrier
! CHECK-NOT:         omp.taskyield
! CHECK:             omp.terminator
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_begin_construct_selected_parent()
  !$omp target
    !$omp begin metadirective &
    !$omp & when(implementation={vendor(llvm)}: parallel)
      !$omp metadirective &
      !$omp & when(construct={target, parallel}: barrier) &
      !$omp & default(taskyield)
    !$omp end metadirective
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_combined_directive(
! CHECK:         omp.target
! CHECK:           omp.parallel
! CHECK-NOT:         omp.taskyield
! CHECK:             omp.barrier
! CHECK:             omp.terminator
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_construct_combined_directive()
  !$omp target
    !$omp parallel
      !$omp metadirective &
      !$omp & when(construct={parallel}: taskyield) &
      !$omp & when(construct={target parallel}: barrier) &
      !$omp & default(nothing)
    !$omp end parallel
  !$omp end target
end subroutine

! The construct context starts at the innermost enclosing TARGET.
! CHECK-LABEL: func.func @_QPtest_construct_outside_target()
! CHECK:         omp.parallel
! CHECK:           omp.target
! CHECK-NOT:         omp.barrier
! CHECK:             omp.taskyield
! CHECK:             omp.terminator
! CHECK:         return
subroutine test_construct_outside_target()
  !$omp parallel
    !$omp target
      !$omp metadirective &
      !$omp & when(construct={parallel}: barrier) &
      !$omp & default(taskyield)
    !$omp end target
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_inside_target()
! CHECK:         omp.parallel
! CHECK:           omp.target
! CHECK:             omp.parallel
! CHECK-NOT:           omp.barrier
! CHECK:               omp.taskwait
! CHECK:               omp.terminator
! CHECK:         return
subroutine test_construct_inside_target()
  !$omp parallel
    !$omp target
      !$omp parallel
        !$omp metadirective &
        !$omp & when(construct={parallel, target}, &
        !$omp &      implementation={vendor(score(10): llvm)}: barrier) &
        !$omp & when(construct={target, parallel}: taskwait) &
        !$omp & default(taskyield)
      !$omp end parallel
    !$omp end target
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_after_target()
! CHECK:         omp.parallel
! CHECK:           omp.target
! CHECK:           omp.barrier
! CHECK-NOT:       omp.taskyield
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_construct_after_target()
  !$omp parallel
    !$omp target
    !$omp end target
    !$omp metadirective &
    !$omp & when(construct={parallel}: barrier) &
    !$omp & default(taskyield)
  !$omp end parallel
end subroutine

! Both candidates score 8. The first selector is a strict subset of the second
! because the second repeats PARALLEL.
! CHECK-LABEL: func.func @_QPtest_repeated_construct_subset()
! CHECK:         omp.wsloop
! CHECK:           omp.parallel
! CHECK-NOT:         omp.taskyield
! CHECK:             omp.taskwait
! CHECK:         return
subroutine test_repeated_construct_subset()
  integer :: i
  !$omp parallel do
  do i = 1, 2
    !$omp parallel
      !$omp metadirective &
      !$omp & when(construct={parallel do}, &
      !$omp &      implementation={vendor(score(4): llvm)}: taskyield) &
      !$omp & when(construct={parallel do, parallel}, &
      !$omp &      implementation={vendor(llvm)}: taskwait)
    !$omp end parallel
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtest_repeated_construct_superset_first()
! CHECK:         omp.wsloop
! CHECK:           omp.parallel
! CHECK-NOT:         omp.taskyield
! CHECK:             omp.taskwait
! CHECK:         return
subroutine test_repeated_construct_superset_first()
  integer :: i
  !$omp parallel do
  do i = 1, 2
    !$omp parallel
      !$omp metadirective &
      !$omp & when(construct={parallel do, parallel}, &
      !$omp &      implementation={vendor(llvm)}: taskwait) &
      !$omp & when(construct={parallel do}, &
      !$omp &      implementation={vendor(score(4): llvm)}: taskyield)
    !$omp end parallel
  end do
  !$omp end parallel do
end subroutine

! The conditions differ, so neither selector is a subset and `a` is tested
! first.
! CHECK-LABEL: func.func @_QPtest_repeated_construct_conditions(
! CHECK-SAME:    %[[ARG0:[^,]*]]: !fir.ref<!fir.logical<4>>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         omp.wsloop
! CHECK:           omp.parallel
! CHECK:             %[[LOADA:.*]] = fir.load %[[A]]#0
! CHECK:             %[[CONDA:.*]] = fir.convert %[[LOADA]]
! CHECK:             fir.if %[[CONDA]] {
! CHECK-NEXT:          omp.taskyield
! CHECK-NEXT:        } else {
! CHECK:               omp.taskwait
! CHECK:         return
subroutine test_repeated_construct_conditions(a, b)
  logical :: a, b
  integer :: i
  !$omp parallel do
  do i = 1, 2
    !$omp parallel
      !$omp metadirective &
      !$omp & when(construct={parallel do}, user={condition(a)}, &
      !$omp &      implementation={vendor(score(4): llvm)}: taskyield) &
      !$omp & when(construct={parallel do, parallel}, user={condition(b)}, &
      !$omp &      implementation={vendor(llvm)}: taskwait)
    !$omp end parallel
  end do
  !$omp end parallel do
end subroutine
