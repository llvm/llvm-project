! Ensure argument -fdebug-dump-provenance works as expected.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang-new -fdebug-dump-provenance %s  2>&1 | FileCheck %s --check-prefix=FLANG

!----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!----------------------------------------
! RUN: %flang-new -fc1 -fdebug-dump-provenance %s  2>&1 | FileCheck %s --check-prefix=FRONTEND

!----------------------------------
! EXPECTED OUTPUT WITH `flang-new`
!----------------------------------
! FLANG:warning: argument unused during compilation: '-fdebug-dump-provenance'

!---------------------------------------
! EXPECTED OUTPUT WITH `flang-new -fc1`
!---------------------------------------
! FRONTEND:AllSources:
! FRONTEND:CookedSource::provenanceMap_:
program A
end
