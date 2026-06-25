! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=STRICT %s
! RUN: %flang_fc1 -fsyntax-only -fno-system-clock-strict %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only -fsystem-clock-strict %s 2>&1 | FileCheck --check-prefix=STRICT %s

! RUN: %flang_fc1 -fsyntax-only -fno-system-clock-strict -fsystem-clock-strict %s 2>&1 | FileCheck --check-prefix=STRICT %s
! RUN: %flang_fc1 -fsyntax-only -fsystem-clock-strict -fno-system-clock-strict %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s

! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 %s 2>&1 | FileCheck --check-prefix=STRICT-8 %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -fno-system-clock-strict %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s

! RUN: %flang_fc1 -fsyntax-only -std=f2018 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only -std=f2023 %s 2>&1 | FileCheck --check-prefix=STRICT %s
! RUN: %flang_fc1 -fsyntax-only -std=f202Y %s 2>&1 | FileCheck --check-prefix=STRICT %s

! RUN: %flang_fc1 -fsyntax-only -std=f2023 -std=f2018 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only -std=f2018 -std=f2023 %s 2>&1 | FileCheck --check-prefix=STRICT %s

! RUN: %flang_fc1 -fsyntax-only -std=f2023 -fno-system-clock-strict %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only -fno-system-clock-strict -std=f2023 %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only -std=f2018 -fsystem-clock-strict %s 2>&1 | FileCheck --check-prefix=STRICT %s
! RUN: %flang_fc1 -fsyntax-only -fsystem-clock-strict -std=f2018 %s 2>&1 | FileCheck --check-prefix=STRICT %s

! Tests for SYSTEM_CLOCK argument warnings

program test_system_clock
  implicit none

  integer(8) :: count8, rate8, max8
  integer(4) :: count4, rate4, max4
  integer(2) :: count2, rate2, max2
  integer(1) :: count1, rate1, max1
  real(8) :: rate_real8
  real(4) :: rate_real4

  call system_clock()

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4)

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate4)

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate_real4)

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate4, max4)

  call system_clock(count8, rate8, max8)

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count=count4)

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count_rate=rate4)

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count_max=max4)

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate_real4, max4)

  call system_clock(count8, rate_real8, max8)

  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count=count4, count_rate=rate4, count_max=max4)

  !STRICT: Integer arguments to SYSTEM_CLOCK should have the same kind.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate8)

  !STRICT: Integer arguments to SYSTEM_CLOCK should have the same kind.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate8, max4)

  !STRICT: Integer arguments to SYSTEM_CLOCK should have the same kind.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate4, max8)

  !STRICT: Integer arguments to SYSTEM_CLOCK should have the same kind.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate_real4, max8)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2, rate2)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2, rate_real4)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2, rate2, max2)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2, rate_real4, max2)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1, rate1)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1, rate_real4)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1, rate1, max1)

  !STRICT: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !STRICT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1, rate_real4, max1)
end program
