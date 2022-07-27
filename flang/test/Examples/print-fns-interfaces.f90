! Check the Flang Print Function Names example plugin doesn't count/print Functions/Subroutines in interfaces
! (It should only count definitions, which will appear elsewhere for interfaced functions/subroutines)
! This requires that the examples are built (FLANG_BUILD_EXAMPLES=ON) to access flangPrintFunctionNames.so

! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangPrintFunctionNames%pluginext -plugin print-fns %s 2>&1 | FileCheck %s

! CHECK: ==== Functions: 0 ====
! CHECK-NEXT: ==== Subroutines: 0 ====

program main
    interface
        function interface_func()
        end function

        subroutine interface_subr()
        end subroutine
    end interface
end program main
