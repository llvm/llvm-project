! REQUIRES: openmp_runtime
! RUN: %not_todo_cmd %flang_fc1 -cpp -DBARE -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -cpp -DTRAITS -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -cpp -DMEMSPACE -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -cpp -DMEMSPACE_TRAITS -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -cpp -DTRAITS_MEMSPACE -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -cpp -DLEGACY -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! Every shape that semantics accepts must reach the established "not yet
! implemented" lowering boundary for USES_ALLOCATORS. Each shape needs its own
! compilation because lowering stops at the first unimplemented clause.

! CHECK: not yet implemented: USES_ALLOCATORS clause is not implemented yet
program p
  use omp_lib
  integer(omp_allocator_handle_kind) :: my_alloc
  type(omp_alloctrait), parameter :: tr(1) = &
      [omp_alloctrait(omp_atk_alignment, 64)]
  integer :: x

#ifdef BARE
  !$omp target uses_allocators(my_alloc)
#endif
#ifdef TRAITS
  !$omp target uses_allocators(traits(tr): my_alloc)
#endif
#ifdef MEMSPACE
  !$omp target uses_allocators(memspace(omp_high_bw_mem_space): my_alloc)
#endif
#ifdef MEMSPACE_TRAITS
  !$omp target uses_allocators(memspace(omp_const_mem_space), traits(tr): my_alloc)
#endif
#ifdef TRAITS_MEMSPACE
  !$omp target uses_allocators(traits(tr), memspace(omp_const_mem_space): my_alloc)
#endif
#ifdef LEGACY
  !$omp target uses_allocators(my_alloc(tr))
#endif
  x = 1
  !$omp end target
end program p
