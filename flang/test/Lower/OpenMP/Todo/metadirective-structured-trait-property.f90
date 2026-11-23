! RUN: %not_todo_cmd %flang_fc1 -cpp -DCLAUSE_PROPERTY -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -cpp -DEXTENSION_PROPERTY -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: clause or extension trait matching in METADIRECTIVE

#ifdef CLAUSE_PROPERTY
subroutine test_construct_simd_clause_property()
  !$omp metadirective &
  !$omp & when(construct={simd(simdlen(8))}: barrier) &
  !$omp & default(nothing)
end subroutine
#endif

#ifdef EXTENSION_PROPERTY
subroutine test_implementation_extension_property()
  !$omp metadirective &
  !$omp & when(implementation={my_trait(foo(bar))}: barrier) &
  !$omp & default(nothing)
end subroutine
#endif
