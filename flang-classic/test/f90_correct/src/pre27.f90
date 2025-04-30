!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests PGIF or FLANG preprocessor macros, depending on whether you are running pgf* compilers or flang compiler.

module check_macro
    implicit none
    contains
    function check_range(macro_name, macro_value, minimum, maximum) result(ret)
        ! Check if input is greater than minumum and less than maximum, both inclusive.
        character(len=*) :: macro_name
        integer :: macro_value, minimum, maximum
        integer :: ret

        if (macro_value .ge. minimum .AND. macro_value .le. maximum) then
            ret = 1
        else
            print *, "FAILED: ", macro_name, " is", macro_value,&
                     &", should be between ", minimum, " and ", maximum
            ret = 0
        endif
    endfunction


    function check_min(macro_name, macro_value, minimum) result(ret)
        ! Check if input is greater than minumum, inclusive.
        character(len=*) :: macro_name
        integer :: macro_value, minimum
        integer :: ret

        if (macro_value .ge. minimum) then
            ret = 1
        else
            print *, "FAILED: ", macro_name, " is", macro_value,&
                     &", should be >= ", minimum
            ret = 0
        endif
    endfunction
endmodule


program p
    use check_macro
    implicit none
    integer :: results(6) = (/0, 0, 0, 0, 0, 0/)
    integer :: expect(6) = (/1, 1, 1, 1, 1, 1/)
 
    #if defined(__PGI) && !defined(__FLANG)
        #ifdef __PGIF90__
            results(1) = check_range("__PGIF90__", __PGIF90__, 1, 99)
        #else
            print *, "FAILED: __PGIF90__ is not defined!"
        #endif
        #ifdef __PGIF90_MINOR__
            results(2) = check_range("__PGIF90_MINOR__", __PGIF90_MINOR__, 1, 99)
        #else
            print *, "FAILED: __PGIF90_MINOR__ is not defined!"
        #endif
        #ifdef __PGIF90_PATCHLEVEL__
            results(3) = check_min("__PGIF90_PATCHLEVEL__", __PGIF90_PATCHLEVEL__, 0)
        #else
            print *, "FAILED: __PGIF90_PATCHLEVEL__ is not defined!"
        #endif

        ! Check that PGIC macros exist.
        #ifdef __PGIC__
            results(4) = check_range("__PGIC__", __PGIC__, 1, 99)
        #else
            print *, "FAILED: __PGIC__ is not defined!"
        #endif
        #ifdef __PGIC_MINOR__
            results(5) = check_range("__PGIC_MINOR__", __PGIC_MINOR__, 1, 99)
        #else
            print *, "FAILED: __PGIC_MINOR__ is not defined!"
        #endif
        #ifdef __PGIC_PATCHLEVEL__
            results(6) = check_min("__PGIC_PATCHLEVEL__", __PGIC_PATCHLEVEL__, 0)
        #else
            print *, "FAILED: __PGIC_PATCHLEVEL__ is not defined!"
        #endif

    #elif defined(__FLANG) && !defined(__PGI)
        #ifdef __FLANG_MAJOR__
            results(1) = check_range("__FLANG_MAJOR__", __FLANG_MAJOR__, 1, 99)
        #else
            print *, "FAILED: __FLANG_MAJOR__ is not defined!"
        #endif
        #ifdef __FLANG_MINOR__
            results(2) = check_range("__FLANG_MINOR__", __FLANG_MINOR__, 1, 99)
        #else
            print *, "FAILED: __FLANG_MINOR__ is not defined!"
        #endif
        #ifdef __FLANG_PATCHLEVEL__
            results(3) = check_min("__FLANG_PATCHLEVEL__", __FLANG_PATCHLEVEL__, 0)
        #else
            print *, "FAILED: __FLANG_PATCHLEVEL__ is not defined!"
        #endif

        ! Check that PGIC macros doesn't exist.
        #ifdef __PGIC__
            print *, "FAILED: __PGIC__ should not be defined!"
        #else
            results(4) = 1
        #endif
        #ifdef __PGIC_MINOR__
            print *, "FAILED: __PGIC_MINOR__ should not be defined!"
        #else
            results(5) = 1
        #endif
        #ifdef __PGIC_PATCHLEVEL__
            print *, "FAILED: __PGIC_PATCHLEVEL__ should not be defined!"
        #else
            results(6) = 1
        #endif
    #elif !defined(__PGI) && !defined(__FLANG)
        print *, "FAILED: Either __PGI or __FLANG should be defined, but neither is!"
    #else
        print *, "FAILED: Only one of __PGI or __FLANG should be defined, not both!"
    #endif

    call check(results, expect, 6)
end program
