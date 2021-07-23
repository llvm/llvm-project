! Ensure argument -I works as expected with an included header.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: not %flang -E %s  2>&1 | FileCheck %s --check-prefix=UNINCLUDED
! RUN: %flang -E -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: %flang -E -I %S/Inputs -I %S/Inputs/header-dir %s  2>&1 | FileCheck %s --check-prefix=MAINDIRECTORY
! RUN: %flang -E -I %S/Inputs/header-dir -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SUBDIRECTORY

!----------------------------------------
! FRONTEND FLANG DRIVER (flang_fc1)
!----------------------------------------
! RUN: not %flang_fc1 -E %s  2>&1 | FileCheck %s --check-prefix=UNINCLUDED
! RUN: %flang_fc1 -E -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: %flang_fc1 -E -I %S/Inputs -I %S/Inputs/header-dir %s  2>&1 | FileCheck %s --check-prefix=MAINDIRECTORY
! RUN: %flang_fc1 -E -I %S/Inputs/header-dir -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SUBDIRECTORY

!--------------------------------------------
! EXPECTED OUTPUT FOR MISSING INCLUDED FILE
!--------------------------------------------
! UNINCLUDED:#include: Source file 'basic-header-one.h' was not found
! UNINCLUDED-NOT:program b
! UNINCLUDED-NOT:program c

!---------------------------------------------
! EXPECTED OUTPUT FOR A SINGLE INCLUDED FOLDER
!--------------------------------------------
! SINGLEINCLUDE:program MainDirectoryOne
! SINGLEINCLUDE-NOT:program X
! SINGLEINCLUDE-NOT:program B
! SINGLEINCLUDE:program MainDirectoryTwo
! SINGLEINCLUDE-NOT:program Y
! SINGLEINCLUDE-NOT:program C

!-------------------------------------------------------
! EXPECTED OUTPUT FOR Inputs/ DIRECTORY SPECIFIED FIRST
!-------------------------------------------------------
! MAINDIRECTORY:program MainDirectoryOne
! MAINDIRECTORY-NOT:program SubDirectoryOne
! MAINDIRECTORY-NOT:program B
! MAINDIRECTORY:program MainDirectoryTwo
! MAINDIRECTORY-NOT:program SubDirectoryTwo
! MAINDIRECTORY-NOT:program C

!------------------------------------------------------------------
! EXPECTED OUTPUT FOR Inputs/header-dir/ DIRECTORY SPECIFIED FIRST
!------------------------------------------------------------------
! SUBDIRECTORY:program SubDirectoryOne
! SUBDIRECTORY-NOT:program MainDirectoryOne
! SUBDIRECTORY-NOT:program B
! SUBDIRECTORY:program SubDirectoryTwo
! SUBDIRECTORY-NOT:program MainDirectoryTwo
! SUBDIRECTORY-NOT:program C

! include-test-one.f90
#include <basic-header-one.h>
#ifdef X
program X
#else
program B
#endif
end

! include-test-two.f90
INCLUDE "basic-header-two.h"
#ifdef Y
program Y
#else
program C
#endif
end
