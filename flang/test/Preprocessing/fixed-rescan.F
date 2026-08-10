! RUN: %flang -E %s | FileCheck %s
! CHECK: call bar
! Ensure that rescanned lines after macro replacement are not
! misinterpreted as fixed-form comments when they start with C or D.
#define foo bar
      call foo
      end
