! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=DEFAULT %s
! RUN: %flang_fc1 -fsyntax-only -Wno-multiple-common-block-init %s 2>&1 | FileCheck --check-prefix=SILENT --allow-empty %s
! RUN: %flang_fc1 -fsyntax-only -w %s 2>&1 | FileCheck --check-prefix=SILENT --allow-empty %s
! RUN: not %flang_fc1 -fsyntax-only -Werror %s 2>&1 | FileCheck --check-prefix=WERROR %s

! Test the -Wmultiple-common-block-init spelling: silencing via
! -Wno-multiple-common-block-init and via blanket -w, and promotion to a
! hard error via blanket -Werror (flang does not support the per-feature
! -Werror=<name> spelling, so blanket -Werror is the strictness mechanism
! for this diagnostic).

subroutine s1
  integer :: i
  common /cw/ i
  data i /1/
end subroutine
subroutine s2
  integer :: i
  common /cw/ i
  data i /1/
end subroutine

! DEFAULT: portability: Multiple initialization of COMMON block /cw/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
! SILENT-NOT: Multiple initialization
! WERROR: error: Semantic errors in {{.*}}common-block-multiple-init-flags.f90
! WERROR: portability: Multiple initialization of COMMON block /cw/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
