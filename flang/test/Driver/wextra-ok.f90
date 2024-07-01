! Ensure that supplying -Wextra into flang-new does not raise error
! The first check should be changed if -Wextra is implemented

! RUN: %flang -std=f2018 -Wextra %s -c 2>&1 | FileCheck %s --check-prefix=CHECK-OK
! RUN: not %flang -std=f2018 -Wblah -Wextra %s -c 2>&1 | FileCheck %s --check-prefix=WRONG

! CHECK-OK: the warning option '-Wextra' is not supported
! WRONG: Only `-Werror` is supported currently.

program wextra_ok
end program wextra_ok
