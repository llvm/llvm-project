! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK-NOT: Character in fixed-form label field must be a digit
#define KWM !
KWM a comment
      end
