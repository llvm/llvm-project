! RUN: %not_todo_cmd %flang_fc1 -cpp -DCLAUSE_PROPERTY -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -cpp -DEXTENSION_PROPERTY -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: clause or extension trait matching in DECLARE VARIANT

#ifdef CLAUSE_PROPERTY
subroutine test_clause_property
  call base_clause()
end subroutine

subroutine base_clause
  interface
    subroutine vsub()
    end subroutine
  end interface
  !$omp declare variant (base_clause:vsub) match (construct={simd(simdlen(8))})
end subroutine
#endif

#ifdef EXTENSION_PROPERTY
subroutine test_extension_property
  call base_ext()
end subroutine

subroutine base_ext
  interface
    subroutine vsub()
    end subroutine
  end interface
  !$omp declare variant (base_ext:vsub) match (implementation={my_trait(foo(bar))})
end subroutine
#endif
