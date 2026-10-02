! Unstructured BLOCKs need independent PFT block mappings in each region.
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DSTATIC %s -o - 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: unstructured associated BLOCK
! CHECK-SAME: in METADIRECTIVE variant
subroutine unstructured_block(flag, selector)
  logical :: flag
  integer :: selector
#ifdef STATIC
  !$omp metadirective when(implementation={vendor(llvm)}: parallel)
#else
  !$omp metadirective when(user={condition(flag)}: parallel) &
  !$omp& otherwise(nothing)
#endif
  block
    go to (10, 20), selector
10  call first_path()
    go to 30
20  call second_path()
30  continue
  end block
end subroutine
