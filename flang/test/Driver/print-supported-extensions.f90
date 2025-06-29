! Test that --print-supported-extensions errors on unsupported architectures.

! REQUIRES: x86-registered-target

! RUN: not %flang --target=x86_64-linux-gnu --print-supported-extensions \
! RUN:   2>&1 | FileCheck %s

! CHECK: error: option '--print-supported-extensions' cannot be specified on this target

end program
