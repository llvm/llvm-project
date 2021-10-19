! Ensure arguments -D and -U work as expected.

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang -E -P %s  2>&1 | FileCheck %s --check-prefix=UNDEFINED
! RUN: %flang -E -P -DX=A %s  2>&1 | FileCheck %s --check-prefix=DEFINED
! RUN: %flang -E -P -DX=A -UX %s  2>&1 | FileCheck %s --check-prefix=UNDEFINED

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: %flang_fc1 -E -P %s  2>&1 | FileCheck %s --check-prefix=UNDEFINED
! RUN: %flang_fc1 -E -P -DX=A %s  2>&1 | FileCheck %s --check-prefix=DEFINED
! RUN: %flang_fc1 -E -P -DX -UX %s  2>&1 | FileCheck %s --check-prefix=UNDEFINED

!--------------------------------------------
! EXPECTED OUTPUT FOR AN UNDEFINED MACRO
!--------------------------------------------
! UNDEFINED:program B
! UNDEFINED-NOT:program X

!--------------------------------------------
! EXPECTED OUTPUT FOR MACRO 'X' DEFINED AS A
!--------------------------------------------
! DEFINED:program A
! DEFINED-NOT:program B

#ifdef X
program X
#else
program B
#endif
end
