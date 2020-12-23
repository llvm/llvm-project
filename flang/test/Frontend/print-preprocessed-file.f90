! Test printpreprocessed action

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang-new -E %s  2>&1 | FileCheck %s

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: %flang-new -fc1 -E %s  2>&1 | FileCheck %s

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! CHECK:program a
! CHECK-NOT:program b
! CHECK-NEXT:x = 1
! CHECK-NEXT:write(*,*) x
! CHECK-NEXT:end

! Preprocessed-file.F:
#define NEW
#ifdef NEW
  program A
#else
  program B
#endif
    x = 1
    write(*,*) x
  end
