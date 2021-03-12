! Ensure argument -fbackslash works as expected.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang-new -E %s  2>&1 | FileCheck %s --check-prefix=ESCAPED
! RUN: %flang-new -E -fbackslash -fno-backslash %s  2>&1 | FileCheck %s --check-prefix=ESCAPED
! RUN: %flang-new -E -fbackslash %s  2>&1 | FileCheck %s --check-prefix=UNESCAPED

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: %flang-new -fc1 -E %s  2>&1 | FileCheck %s --check-prefix=ESCAPED
! RUN: %flang-new -fc1 -E -fbackslash -fno-backslash %s  2>&1 | FileCheck %s --check-prefix=ESCAPED
! RUN: %flang-new -fc1 -E -fbackslash %s  2>&1 | FileCheck %s --check-prefix=UNESCAPED

!-----------------------------------------
! EXPECTED OUTPUT FOR ESCAPED BACKSLASHES
!-----------------------------------------
! ESCAPED:program backslash
! ESCAPED-NEXT:New\\nline
! ESCAPED-NOT:New\nline

!-------------------------------------------
! EXPECTED OUTPUT FOR UNESCAPED BACKSLASHES
!-------------------------------------------
! UNESCAPED:program backslash
! UNESCAPED-NEXT:New\nline
! UNESCAPED-NOT:New\\nline

program Backslash
    print *, 'New\nline'
end
