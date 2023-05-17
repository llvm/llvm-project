! Test the behaviors of -f{no-}color-diagnostics when emitting scanning
! diagnostics.
! Windows command prompt doesn't support ANSI escape sequences.
! REQUIRES: shell

! RUN: not %flang %s -E -Werror -fcolor-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_CD
! RUN: not %flang %s -E -Werror -fno-color-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_NCD
! RUN: not %flang_fc1 -E -Werror %s -fcolor-diagnostics 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK_CD
! RUN: not %flang_fc1 -E -Werror %s 2>&1 | FileCheck %s --check-prefix=CHECK_NCD

! CHECK_CD: {{.*}}[0;1;35mwarning: {{.*}}[0mCharacter in fixed-form label field must be a digit

! CHECK_NCD: warning: Character in fixed-form label field must be a digit

1 continue
end
