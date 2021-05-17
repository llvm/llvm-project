! Ensure argument -fimplicit-none works as expected.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -fsyntax-only %s  2>&1 | FileCheck %s --allow-empty --check-prefix=DEFAULT
! RUN: %flang -fsyntax-only -fimplicit-none -fno-implicit-none %s  2>&1 | FileCheck %s --allow-empty --check-prefix=DEFAULT
! RUN: not %flang -fsyntax-only -fimplicit-none %s  2>&1 | FileCheck %s --check-prefix=WITH_IMPL_NONE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang_fc1)
!-----------------------------------------
! RUN: %flang_fc1 -fsyntax-only %s  2>&1 | FileCheck %s --allow-empty --check-prefix=DEFAULT
! RUN: %flang_fc1 -fsyntax-only -fimplicit-none -fno-implicit-none %s  2>&1 | FileCheck %s --allow-empty --check-prefix=DEFAULT
! RUN: not %flang_fc1 -fsyntax-only -fimplicit-none %s  2>&1 | FileCheck %s --check-prefix=WITH_IMPL_NONE

!--------------------------------------
! EXPECTED OUTPUT FOR NO IMPLICIT NONE
!--------------------------------------
! DEFAULT-NOT:error

!------------------------------------------
! EXPECTED OUTPUT FOR IMPLICIT NONE ALWAYS
!------------------------------------------
! WITH_IMPL_NONE:No explicit type declared for 'a'
! WITH_IMPL_NONE:No explicit type declared for 'b'

function a()
  a = b
end
