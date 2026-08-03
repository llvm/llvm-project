! A metadirective block replacement cannot yet reconstruct implicit
! data-sharing attributes. Reject regions that would otherwise lose a
! sequential loop IV or an implicit task capture.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/parallel.f90 2>&1 | FileCheck --check-prefix=TODO %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/task.f90 2>&1 | FileCheck --check-prefix=TODO %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/task-unselected-shared.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=TODO %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/task-nested-shared.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=TODO %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/task-default-shared-sequential-loop.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=TODO %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/task-local.f90 | FileCheck --check-prefix=LOCAL %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/task-selected-shared.f90 \
! RUN:   | FileCheck --check-prefix=SHARED %s

! TODO: not yet implemented: data-environment construct in METADIRECTIVE variant

! LOCAL-LABEL: func.func @_QPtask_block_variant_with_local_storage
! LOCAL:         omp.task {
! LOCAL:           fir.alloca i32 {{.*}}bindc_name = "local"
! LOCAL:           hlfir.assign
! LOCAL:           omp.terminator

! SHARED-LABEL: func.func @_QPtask_block_variant_with_selected_shared
! SHARED:         %[[EXPLICIT:.*]]:2 = hlfir.declare
! SHARED:         omp.task {
! SHARED:           fir.load %[[EXPLICIT]]#0
! SHARED:           hlfir.assign {{.*}} to %[[EXPLICIT]]#0
! SHARED:           omp.terminator

! SHARED-LABEL: func.func @_QPtask_block_variant_with_default_shared
! SHARED:         %[[DEFAULT:.*]]:2 = hlfir.declare
! SHARED:         omp.task {
! SHARED:           fir.load %[[DEFAULT]]#0
! SHARED:           hlfir.assign {{.*}} to %[[DEFAULT]]#0
! SHARED:           omp.terminator

!--- parallel.f90
subroutine parallel_block_variant(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel) &
  !$omp& otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

!--- task.f90
subroutine task_block_variant(x)
  integer :: x
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: task) &
  !$omp& otherwise(nothing)
  x = x + 1
  !$omp end metadirective
end subroutine

!--- task-unselected-shared.f90
subroutine task_block_variant_with_unselected_shared(x)
  integer :: x
  !$omp begin metadirective &
  !$omp& when(user={condition(score(2): .true.)}: task) &
  !$omp& when(user={condition(score(1): .true.)}: task shared(x)) &
  !$omp& otherwise(nothing)
  x = x + 1
  !$omp end metadirective
end subroutine

!--- task-nested-shared.f90
subroutine task_block_variant_with_nested_shared(x)
  integer :: x
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: task) &
  !$omp& otherwise(nothing)
  !$omp parallel shared(x)
  x = x + 1
  !$omp end parallel
  !$omp end metadirective
end subroutine

!--- task-default-shared-sequential-loop.f90
subroutine task_block_variant_with_default_shared_sequential_loop(n)
  integer :: n, i
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: task default(shared)) &
  !$omp& otherwise(nothing)
  do i = 1, n
  end do
  !$omp end metadirective
end subroutine

!--- task-local.f90
subroutine task_block_variant_with_local_storage()
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: task) &
  !$omp& otherwise(nothing)
  block
    integer :: local
    local = 1
  end block
  !$omp end metadirective
end subroutine

!--- task-selected-shared.f90
subroutine task_block_variant_with_selected_shared(x)
  integer :: x
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: task shared(x)) &
  !$omp& otherwise(nothing)
  x = x + 1
  !$omp end metadirective
end subroutine

subroutine task_block_variant_with_default_shared(x)
  integer :: x
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: task default(shared)) &
  !$omp& otherwise(nothing)
  x = x + 1
  !$omp end metadirective
end subroutine
