!RUN: not %flang -fsyntax-only %s 2>&1 | FileCheck %s
!CHECK: bug50563.f:3:1: error: Character in fixed-form label field must be a digit
pi=3
      end
