! RUN: split-file %s %t
! RUN: not %flang_fc1 -fsyntax-only -fopenmp %t/stray_end1.f90 2>&1 | FileCheck %t/stray_end1.f90
! RUN: not %flang_fc1 -fsyntax-only -fopenmp %t/stray_end2.f90 2>&1 | FileCheck %t/stray_end2.f90
! RUN: not %flang_fc1 -fsyntax-only -fopenmp %t/stray_begin.f90 2>&1 | FileCheck %t/stray_begin.f90


!--- stray_end1.f90
! Parser error

subroutine stray_end1
  !CHECK: error: Misplaced OpenMP end-directive
  !$omp end tile
end subroutine


!--- stray_end2.f90

subroutine stray_end2
  print *
  !CHECK: error: Misplaced OpenMP end-directive
  !$omp end tile
end subroutine


!--- stray_begin.f90

subroutine stray_begin
  !CHECK: error: OpenMP loop construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp tile sizes(2)
end subroutine

