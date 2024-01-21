! UNSUPPORTED: system-windows, system-darwin, system-aix

! RUN: %flang -x ir -o %t.c-object -c %S/Inputs/no_duplicate_main.ll
! RUN: %flang -o %t -c %s
! RUN: not %flang -o %t.exe %t %t.c-object 2>&1

! RUN: %flang -fno-fortran-main -o %t.exe %t %t.c-object 2>&1

! TODO: potentially add further checks to ensure that proper
!       linker error messages are detected and checked via
!       FileCheck.

program main_dupes
    ! Irrelevant what to do in here.
    ! Test is supposed to fail at link time.
end program main_dupes
