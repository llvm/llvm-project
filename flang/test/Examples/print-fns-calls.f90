! Check the Flang Print Function Names example plugin doesn't count/print function/subroutine calls (should only count definitions)
! This requires that the examples are built (FLANG_BUILD_EXAMPLES=ON) to access flangPrintFunctionNames.so

! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangPrintFunctionNames%pluginext -plugin print-fns %s 2>&1 | FileCheck %s

!-----------------------------
! EXPECTED OUTPUT: Counts == 0
!-----------------------------
! CHECK: ==== Functions: 0 ====
! CHECK-NEXT: ==== Subroutines: 0 ====

!-----------------------------
! INPUT
!-----------------------------
program main
    call subroutine1
    fn1 = function1()
    fn2 = function2()
end program main
