! Ensure argument -std=f2018 works as expected.

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: %flang_fc1 -fsyntax-only %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: %flang_fc1 -fsyntax-only -std=f2018 %s  2>&1 | FileCheck %s --check-prefix=GIVEN
! RUN: %flang_fc1 -fsyntax-only -pedantic %s  2>&1 | FileCheck %s --check-prefix=GIVEN

!-----------------------------------------
! EXPECTED OUTPUT WITHOUT
!-----------------------------------------
! WITHOUT-NOT: A DO loop should terminate with an END DO or CONTINUE

!-----------------------------------------
! EXPECTED OUTPUT WITH
!-----------------------------------------
! GIVEN: A DO loop should terminate with an END DO or CONTINUE

subroutine foo2()
    do 01 m=1,2
      select case (m)
      case default
        print*, "default", m
      case (1)
        print*, "start"
01    end select
end subroutine
