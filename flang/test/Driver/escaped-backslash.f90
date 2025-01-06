! Ensure argument -fbackslash works as expected.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -E %s  2>&1 | FileCheck %s --check-prefix=ESCAPED
! RUN: %flang -E -fbackslash -fno-backslash %s  2>&1 | FileCheck %s --check-prefix=ESCAPED
! RUN: %flang -E -fbackslash %s  2>&1 | FileCheck %s --check-prefix=UNESCAPED

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!-----------------------------------------
! RUN: %flang_fc1 -E %s  2>&1 | FileCheck %s --check-prefix=ESCAPED
! RUN: %flang_fc1 -E -fbackslash -fno-backslash %s  2>&1 | FileCheck %s --check-prefix=ESCAPED
! RUN: %flang_fc1 -E -fbackslash %s  2>&1 | FileCheck %s --check-prefix=UNESCAPED

! ESCAPED:program Backslash
! ESCAPED-NEXT:New\\nline
! ESCAPED-NOT:New\nline

! UNESCAPED:program Backslash
! UNESCAPED-NEXT:New\nline
! UNESCAPED-NOT:New\\nline

program Backslash
    print *, 'New\nline'
end
