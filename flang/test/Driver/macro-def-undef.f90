! Ensure arguments -D and -U work as expected.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang-new -E %s  2>&1 | FileCheck %s --check-prefix=UNDEFINED
! RUN: %flang-new -E -DX=A %s  2>&1 | FileCheck %s --check-prefix=DEFINED
! RUN: %flang-new -E -DX=A -UX %s  2>&1 | FileCheck %s --check-prefix=UNDEFINED

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: %flang-new -fc1 -E %s  2>&1 | FileCheck %s --check-prefix=UNDEFINED
! RUN: %flang-new -fc1 -E -DX=A %s  2>&1 | FileCheck %s --check-prefix=DEFINED
! RUN: %flang-new -fc1 -E -DX -UX %s  2>&1 | FileCheck %s --check-prefix=UNDEFINED

!--------------------------------------------
! EXPECTED OUTPUT FOR AN UNDEFINED MACRO
!--------------------------------------------
! UNDEFINED:program b
! UNDEFINED-NOT:program x
! UNDEFINED-NEXT:end

!--------------------------------------------
! EXPECTED OUTPUT FOR MACRO 'X' DEFINED AS A
!--------------------------------------------
! DEFINED:program a
! DEFINED-NOT:program b
! DEFINED-NEXT:end

#ifdef X
program X
#else
program B
#endif
end