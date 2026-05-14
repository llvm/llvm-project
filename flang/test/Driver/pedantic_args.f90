! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --no-pedantic %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f77 --no-pedantic %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f77 %s 2>&1 | FileCheck --check-prefix=F77-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f1977 %s 2>&1 | FileCheck --check-prefix=F77-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f90 %s 2>&1 | FileCheck --check-prefix=F90-F95-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f1990 %s 2>&1 | FileCheck --check-prefix=F90-F95-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f95 %s 2>&1 | FileCheck --check-prefix=F90-F95-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f1995 %s 2>&1 | FileCheck --check-prefix=F90-F95-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f2003 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f03 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f2008 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f08 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f2018 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f18 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f2023 %s 2>&1 | FileCheck --check-prefix=F2023-F202Y-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f23 %s 2>&1 | FileCheck --check-prefix=F2023-F202Y-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f202Y %s 2>&1 | FileCheck --check-prefix=F2023-F202Y-WARNING %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f2Y %s 2>&1 | FileCheck --check-prefix=F2023-F202Y-WARNING %s

! Tests for arguments to -pedantic flag, using SYSTEM_CLOCK argument warnings

program test_system_clock
  implicit none

  integer(8) :: max8
  integer(2) :: count2
  real(4) :: rate_real4

  !F77-WARNING: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !F90-F95-WARNING: Argument to SYSTEM_CLOCK should be integer.
  !F90-F95-WARNING: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !F2023-F202Y-WARNING: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !F2023-F202Y-WARNING: Integer arguments to SYSTEM_CLOCK should have the same kind.
  call system_clock(count2, rate_real4, max8)

end program
