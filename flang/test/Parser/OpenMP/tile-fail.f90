! RUN: split-file %s %t
! RUN: not %flang_fc1 -fsyntax-only -fopenmp %t/stray_end1.f90 2>&1 | FileCheck %t/stray_end1.f90
! RUN: not %flang_fc1 -fsyntax-only -fopenmp %t/stray_end2.f90 2>&1 | FileCheck %t/stray_end2.f90
! RUN: not %flang_fc1 -fsyntax-only -fopenmp %t/stray_begin.f90 2>&1 | FileCheck %t/stray_begin.f90


!--- stray_end1.f90
! Parser error

subroutine stray_end1
  !CHECK: error: expected OpenMP construct
  !$omp end tile
end subroutine


!--- stray_end2.f90
! Semantic error

subroutine stray_end2
  print *
  !CHECK: error: The END TILE directive must follow the DO loop associated with the loop construct
  !$omp end tile
end subroutine


!--- stray_begin.f90

subroutine stray_begin
  !CHECK: error: A DO loop must follow the TILE directive
  !$omp tile sizes(2)
end subroutine

