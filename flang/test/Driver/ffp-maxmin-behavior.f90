! Test that -ffp-maxmin-behavior is accepted by flang -fc1 (all values and
! unknown defaulting to legacy) and is not recognized by the flang driver.

program p
end program p

! flang -fc1 accepts all valid values
! RUN: %flang_fc1 -fsyntax-only -ffp-maxmin-behavior=legacy %s
! RUN: %flang_fc1 -fsyntax-only -ffp-maxmin-behavior=portable %s
! RUN: %flang_fc1 -fsyntax-only -ffp-maxmin-behavior=extremum %s
! RUN: %flang_fc1 -fsyntax-only -ffp-maxmin-behavior=extremenum %s

! flang -fc1 accepts unknown value (defaults to legacy, no error)
! RUN: %flang_fc1 -fsyntax-only -ffp-maxmin-behavior=invalid %s

! flang driver does not forward the option to -fc1 (fc1-only option)
! RUN: not %flang -### -ffp-maxmin-behavior=legacy %s 2>&1 \
! RUN:   | FileCheck %s -check-prefix=DRIVER-NO-FORWARD
! DRIVER-NO-FORWARD: error: unknown argument '-ffp-maxmin-behavior=legacy'
! DRIVER-NO-FORWARD: "-fc1"
! DRIVER-NO-FORWARD-NOT: -ffp-maxmin-behavior
