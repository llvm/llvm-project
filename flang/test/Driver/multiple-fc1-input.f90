! Test that flang -fc1 can be called with several input files without
! crashing.
! Regression tests for: https://github.com/llvm/llvm-project/issues/137126

! RUN: %flang_fc1 -emit-fir %s %s -o - | FileCheck %s
subroutine foo()
end subroutine
! CHECK: func @_QPfoo() 
! CHECK: func @_QPfoo() 
