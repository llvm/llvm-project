! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: loop-associated METADIRECTIVE without associated DO

module m
  !$omp metadirective when(implementation={vendor(llvm)}: do) default(nothing)
end module
