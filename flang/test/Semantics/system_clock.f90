! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --allow-empty --implicit-check-not="SYSTEM_CLOCK" %s
! RUN: %flang_fc1 -fsyntax-only -Wsystem-clock-not-intrinsic %s 2>&1 | FileCheck --check-prefix=NOT-INTRINSIC %s
! RUN: %flang_fc1 -fsyntax-only -Wsystem-clock-args-only-integer %s 2>&1 | FileCheck --check-prefix=INTEGER-ONLY %s
! RUN: %flang_fc1 -fsyntax-only -Wsystem-clock-int-args-only-default %s 2>&1 | FileCheck --check-prefix=DEFAULT-ONLY %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -Wsystem-clock-int-args-only-default %s 2>&1 | FileCheck --check-prefix=DEFAULT-8 %s
! RUN: %flang_fc1 -fsyntax-only -Wsystem-clock-int-args-same-kind %s 2>&1 | FileCheck --check-prefix=ARG-KIND-MISMATCH %s
! RUN: %flang_fc1 -fsyntax-only -Wsystem-clock-min-size %s 2>&1 | FileCheck --check-prefix=MIN-SIZE %s
! RUN: %flang_fc1 -fsyntax-only -fdefault-integer-8 -Wsystem-clock-min-size %s 2>&1 | FileCheck --check-prefix=MIN-SIZE-8 %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f77 %s 2>&1 | FileCheck --check-prefix=NOT-INTRINSIC %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f90 %s 2>&1 | FileCheck --check-prefix=INTEGER-ONLY,DEFAULT-ONLY %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f95 %s 2>&1 | FileCheck --check-prefix=INTEGER-ONLY,DEFAULT-ONLY %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f2023 %s 2>&1 | FileCheck --check-prefix=ARG-KIND-MISMATCH,MIN-SIZE %s
! RUN: %flang_fc1 -fsyntax-only --pedantic=f202Y %s 2>&1 | FileCheck --check-prefix=ARG-KIND-MISMATCH,MIN-SIZE %s

! Tests for SYSTEM_CLOCK argument warnings

program test_system_clock
  implicit none

  integer(8) :: count8, rate8, max8
  integer(4) :: count4, rate4, max4
  integer(2) :: count2, rate2, max2
  integer(1) :: count1, rate1, max1
  real(8) :: rate_real8
  real(4) :: rate_real4

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  call system_clock()

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !INTEGER-ONLY: Argument to SYSTEM_CLOCK should be integer.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate_real4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate4, max4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  call system_clock(count8, rate8, max8)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count=count4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count_rate=rate4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count_max=max4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !INTEGER-ONLY: Argument to SYSTEM_CLOCK should be integer.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate_real4, max4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !INTEGER-ONLY: Argument to SYSTEM_CLOCK should be integer.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  call system_clock(count8, rate_real8, max8)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count=count4, count_rate=rate4, count_max=max4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !ARG-KIND-MISMATCH: Integer arguments to SYSTEM_CLOCK should have the same kind.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate8)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !ARG-KIND-MISMATCH: Integer arguments to SYSTEM_CLOCK should have the same kind.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate8, max4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !ARG-KIND-MISMATCH: Integer arguments to SYSTEM_CLOCK should have the same kind.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate4, max8)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !INTEGER-ONLY: Argument to SYSTEM_CLOCK should be integer.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !ARG-KIND-MISMATCH: Integer arguments to SYSTEM_CLOCK should have the same kind.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count4, rate_real4, max8)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2, rate2)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !INTEGER-ONLY: Argument to SYSTEM_CLOCK should be integer.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2, rate_real4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2, rate2, max2)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !INTEGER-ONLY: Argument to SYSTEM_CLOCK should be integer.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count2, rate_real4, max2)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1, rate1)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !INTEGER-ONLY: Argument to SYSTEM_CLOCK should be integer.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1, rate_real4)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1, rate1, max1)

  !NOT-INTRINSIC: Intrinsic SYSTEM_CLOCK is not part of the Fortran 77 standard.
  !INTEGER-ONLY: Argument to SYSTEM_CLOCK should be integer.
  !DEFAULT-ONLY: Integer argument to SYSTEM_CLOCK should be an integer with kind == 4.
  !DEFAULT-8: Integer argument to SYSTEM_CLOCK should be an integer with kind == 8.
  !MIN-SIZE: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 4.
  !MIN-SIZE-8: Integer argument to SYSTEM_CLOCK should be an integer with kind >= 8.
  call system_clock(count1, rate_real4, max1)
end program
