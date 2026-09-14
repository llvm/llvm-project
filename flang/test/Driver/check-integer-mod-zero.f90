! Test driver handling of -fcheck-integer-mod-zero.

! RUN: %flang -fcheck-integer-mod-zero -### %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ENABLE
! ENABLE: "-fc1"
! ENABLE-SAME: "-fcheck-integer-mod-zero"

! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
! DEFAULT: "-fc1"
! DEFAULT-NOT: "check-integer-mod-zero"

end
