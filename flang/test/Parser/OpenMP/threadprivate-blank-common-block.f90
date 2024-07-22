! RUN: not %flang_fc1 -fsyntax-only %s -fopenmp 2>&1 | FileCheck %s
! From Standard: A blank common block cannot appear in a threadprivate directive.

program main
    integer :: a
    common//a
    !CHECK: error: expected one of '$@ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    !$omp threadprivate(//)
 end
