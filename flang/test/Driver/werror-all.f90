! Ensures that -Werror is read regardless of whether or not other -W
! flags are present in the CLI args

! RUN: not %flang -std=f2018 -Werror -Wextra %s 2>&1 | FileCheck %s --check-prefix=WRONG
! RUN: %flang -std=f2018 -Wextra -Wall %s 2>&1 | FileCheck %s --check-prefix=CHECK-OK

! WRONG: Semantic errors in
! CHECK-OK: FORALL index variable

program werror_check_all
  integer :: a(3)
  forall (j=1:n) a(i) = 1
end program werror_check_all
