! A BLOCK has no data-sharing attributes for a metadirective-selected data
! environment, so its sequential DO variable would not be privatized.
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DPARALLEL %s -o - 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DTASK %s -o - 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DTEAMS %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: data-environment construct with associated BLOCK
! CHECK-SAME: in METADIRECTIVE variant
subroutine block_data_environment(flag)
  logical :: flag
  integer :: i
#if defined(PARALLEL)
  !$omp metadirective when(implementation={vendor(llvm)}: parallel)
#elif defined(TASK)
  !$omp metadirective when(user={condition(flag)}: task) otherwise(nothing)
#else
  !$omp metadirective when(implementation={vendor(llvm)}: teams)
#endif
  block
    do i = 1, 2
      call observe(i)
    end do
  end block
end subroutine
